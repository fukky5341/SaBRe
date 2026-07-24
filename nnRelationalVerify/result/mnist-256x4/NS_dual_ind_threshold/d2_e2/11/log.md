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
execution time: IAR + RelationalAnalysis = 1.75 + 2.88 = 4.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0125356, upper bound: 0.0125359

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124479, upper bound: 0.0125331
time: 2.30 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0125068, upper bound: 0.0125069
time: 1.80 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.24 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.24
Output dim: 1, lower bound: -0.0124479, upper bound: 0.0125331
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.24
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

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123591, upper bound: 0.0123929
time: 2.86 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123266, upper bound: 0.0124142
time: 2.58 seconds

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

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0125069, upper bound: 0.0124479
time: 2.34 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0125069, upper bound: 0.0125069
time: 2.06 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 6.12 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 6.12
Output dim: 1, lower bound: -0.0123591, upper bound: 0.0123929
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 6.12
Output dim: 1, lower bound: -0.0123266, upper bound: 0.0124142
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 6.12
Output dim: 1, lower bound: -0.0125069, upper bound: 0.0124479
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 6.12
Output dim: 1, lower bound: -0.0125069, upper bound: 0.0125069

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

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123139, upper bound: 0.0123925
time: 2.48 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123139, upper bound: 0.0123931
time: 2.65 seconds

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

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123139, upper bound: 0.0124136
time: 2.52 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123139, upper bound: 0.0124144
time: 3.01 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0045201, 0.0032881, -0.0034953, 0.0030272, -0.0075472, 0.0067835
1: 0.9881709, 1.0059903, 0.9888287, 1.0039555, -0.0157846, 0.0171615
2: -0.0152950, 0.0035432, -0.0141504, 0.0024819, -0.0172348, 0.0171618
3: 0.0004490, 0.0062502, 0.0007359, 0.0059773, -0.0055283, 0.0055144
4: -0.0044551, 0.0105054, -0.0033131, 0.0096007, -0.0140558, 0.0138185
5: -0.0012789, 0.0117491, -0.0008122, 0.0111097, -0.0123886, 0.0125614
6: -0.0046915, 0.0047088, -0.0040328, 0.0038113, -0.0085028, 0.0087416
7: -0.0120721, -0.0014747, -0.0116117, -0.0022180, -0.0098541, 0.0101369
8: -0.0113280, 0.0183474, -0.0102891, 0.0168435, -0.0280117, 0.0284715
9: -0.0107042, 0.0063085, -0.0098691, 0.0056991, -0.0164033, 0.0161776

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 162

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123918, upper bound: 0.0123583
time: 2.46 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124133, upper bound: 0.0123263
time: 2.62 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0045201, 0.0032881, -0.0045201, 0.0032881, -0.0078082, 0.0078082
1: 0.9881709, 1.0059903, 0.9881709, 1.0059903, -0.0178194, 0.0178194
2: -0.0152950, 0.0035432, -0.0152950, 0.0035432, -0.0182331, 0.0182331
3: 0.0004490, 0.0062502, 0.0004490, 0.0062502, -0.0058012, 0.0058012
4: -0.0044551, 0.0105054, -0.0044551, 0.0105054, -0.0149604, 0.0149604
5: -0.0012789, 0.0117491, -0.0012789, 0.0117491, -0.0130281, 0.0130281
6: -0.0046915, 0.0047088, -0.0046915, 0.0047088, -0.0094003, 0.0094003
7: -0.0120721, -0.0014747, -0.0120721, -0.0014747, -0.0105974, 0.0105974
8: -0.0113280, 0.0183474, -0.0113280, 0.0183474, -0.0294956, 0.0294956
9: -0.0107042, 0.0063085, -0.0107042, 0.0063085, -0.0170127, 0.0170127

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 162

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123918, upper bound: 0.0123591
time: 2.56 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124133, upper bound: 0.0123263
time: 2.12 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.33 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.33
Output dim: 1, lower bound: -0.0123139, upper bound: 0.0123925
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.33
Output dim: 1, lower bound: -0.0123139, upper bound: 0.0123931
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.33
Output dim: 1, lower bound: -0.0123139, upper bound: 0.0124136
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.33
Output dim: 1, lower bound: -0.0123139, upper bound: 0.0124144
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.33
Output dim: 1, lower bound: -0.0123918, upper bound: 0.0123583
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.33
Output dim: 1, lower bound: -0.0124133, upper bound: 0.0123263
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.33
Output dim: 1, lower bound: -0.0123918, upper bound: 0.0123591
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.33
Output dim: 1, lower bound: -0.0124133, upper bound: 0.0123263

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

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123588, upper bound: 0.0123410
time: 2.44 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123588, upper bound: 0.0123926
time: 3.31 seconds

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

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123588, upper bound: 0.0123409
time: 5.74 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123588, upper bound: 0.0123935
time: 2.63 seconds

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

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123136, upper bound: 0.0123559
time: 2.33 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123136, upper bound: 0.0124145
time: 2.28 seconds

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

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123136, upper bound: 0.0123409
time: 2.31 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123136, upper bound: 0.0123926
time: 2.37 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042693, 0.0030647, -0.0034953, 0.0030272, -0.0072965, 0.0065601
1: 0.9887339, 1.0054924, 0.9888287, 1.0039555, -0.0152215, 0.0166637
2: -0.0143153, 0.0032835, -0.0141504, 0.0024819, -0.0162434, 0.0169019
3: 0.0005192, 0.0060166, 0.0007359, 0.0059773, -0.0054581, 0.0052807
4: -0.0037652, 0.0097310, -0.0033131, 0.0096007, -0.0133659, 0.0130442
5: -0.0011647, 0.0112018, -0.0008122, 0.0111097, -0.0122744, 0.0120140
6: -0.0041817, 0.0039406, -0.0040328, 0.0038113, -0.0079931, 0.0079734
7: -0.0116780, -0.0016566, -0.0116117, -0.0022180, -0.0094599, 0.0099550
8: -0.0110738, 0.0170602, -0.0102891, 0.0168435, -0.0277582, 0.0271826
9: -0.0099894, 0.0061594, -0.0098691, 0.0056991, -0.0156884, 0.0160285

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123935, upper bound: 0.0123135
time: 3.55 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123935, upper bound: 0.0123261
time: 2.41 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0057187, 0.0037577, -0.0034671, 0.0030074, -0.0087261, 0.0072249
1: 0.9885953, 1.0083705, 0.9888785, 1.0038997, -0.0153044, 0.0194920
2: -0.0145566, 0.0047846, -0.0140637, 0.0024527, -0.0164930, 0.0183706
3: 0.0001134, 0.0060741, 0.0007438, 0.0059566, -0.0058432, 0.0053304
4: -0.0056728, 0.0099217, -0.0032266, 0.0095322, -0.0152050, 0.0131483
5: -0.0018248, 0.0113366, -0.0007994, 0.0110612, -0.0128860, 0.0121359
6: -0.0071285, 0.0041298, -0.0040027, 0.0037433, -0.0108718, 0.0081325
7: -0.0117751, -0.0006053, -0.0115768, -0.0022385, -0.0095366, 0.0109715
8: -0.0125432, 0.0173771, -0.0102605, 0.0167296, -0.0291255, 0.0274768
9: -0.0101654, 0.0070215, -0.0098058, 0.0056823, -0.0158477, 0.0168272

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123559, upper bound: 0.0123135
time: 3.16 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123559, upper bound: 0.0123263
time: 4.34 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042693, 0.0030647, -0.0045201, 0.0032881, -0.0075574, 0.0075848
1: 0.9887339, 1.0054924, 0.9881709, 1.0059903, -0.0172563, 0.0173216
2: -0.0143153, 0.0032835, -0.0152950, 0.0035432, -0.0172435, 0.0179735
3: 0.0005192, 0.0060166, 0.0004490, 0.0062502, -0.0057310, 0.0055676
4: -0.0037652, 0.0097310, -0.0044551, 0.0105054, -0.0142705, 0.0141861
5: -0.0011647, 0.0112018, -0.0012789, 0.0117491, -0.0129139, 0.0124807
6: -0.0041817, 0.0039406, -0.0046915, 0.0047088, -0.0088906, 0.0086321
7: -0.0116780, -0.0016566, -0.0120721, -0.0014747, -0.0102032, 0.0104155
8: -0.0110738, 0.0170602, -0.0113280, 0.0183474, -0.0292422, 0.0282070
9: -0.0099894, 0.0061594, -0.0107042, 0.0063085, -0.0162979, 0.0168636

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 162

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123918, upper bound: 0.0123136
time: 2.47 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123918, upper bound: 0.0123262
time: 2.65 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0057187, 0.0037577, -0.0044950, 0.0032682, -0.0089869, 0.0082527
1: 0.9885953, 1.0083705, 0.9882210, 1.0059406, -0.0173452, 0.0201495
2: -0.0145566, 0.0047846, -0.0152076, 0.0035172, -0.0174966, 0.0194443
3: 0.0001134, 0.0060741, 0.0004560, 0.0062294, -0.0061159, 0.0056181
4: -0.0056728, 0.0099217, -0.0043679, 0.0104363, -0.0161091, 0.0142896
5: -0.0018248, 0.0113366, -0.0012675, 0.0117003, -0.0135252, 0.0126041
6: -0.0071285, 0.0041298, -0.0046405, 0.0046403, -0.0117688, 0.0087703
7: -0.0117751, -0.0006053, -0.0120370, -0.0014929, -0.0102821, 0.0114317
8: -0.0125432, 0.0173771, -0.0113025, 0.0182326, -0.0306085, 0.0285055
9: -0.0101654, 0.0070215, -0.0106404, 0.0062936, -0.0164590, 0.0176619

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 162

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124136, upper bound: 0.0123136
time: 4.73 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124136, upper bound: 0.0123262
time: 2.03 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 8.37 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.37
Output dim: 1, lower bound: -0.0123588, upper bound: 0.0123410
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.37
Output dim: 1, lower bound: -0.0123588, upper bound: 0.0123926
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.37
Output dim: 1, lower bound: -0.0123588, upper bound: 0.0123409
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.37
Output dim: 1, lower bound: -0.0123588, upper bound: 0.0123935
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.37
Output dim: 1, lower bound: -0.0123136, upper bound: 0.0123559
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.37
Output dim: 1, lower bound: -0.0123136, upper bound: 0.0124145
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.37
Output dim: 1, lower bound: -0.0123136, upper bound: 0.0123409
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.37
Output dim: 1, lower bound: -0.0123136, upper bound: 0.0123926
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.37
Output dim: 1, lower bound: -0.0123935, upper bound: 0.0123135
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.37
Output dim: 1, lower bound: -0.0123935, upper bound: 0.0123261
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.37
Output dim: 1, lower bound: -0.0123559, upper bound: 0.0123135
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.37
Output dim: 1, lower bound: -0.0123559, upper bound: 0.0123263
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.37
Output dim: 1, lower bound: -0.0123918, upper bound: 0.0123136
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.37
Output dim: 1, lower bound: -0.0123918, upper bound: 0.0123262
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.37
Output dim: 1, lower bound: -0.0124136, upper bound: 0.0123136
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.37
Output dim: 1, lower bound: -0.0124136, upper bound: 0.0123262

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

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122500, upper bound: 0.0123850
time: 3.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123457, upper bound: 0.0123926
time: 2.52 seconds

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

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122500, upper bound: 0.0124253
time: 3.54 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123457, upper bound: 0.0124303
time: 2.98 seconds

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

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122377, upper bound: 0.0123046
time: 2.66 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123297, upper bound: 0.0123138
time: 3.28 seconds

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

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122377, upper bound: 0.0123568
time: 3.06 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123297, upper bound: 0.0123648
time: 2.42 seconds

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

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.12 seconds

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
time: 2.80 seconds

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

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122003, upper bound: 0.0124146
time: 4.02 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122857, upper bound: 0.0124203
time: 2.97 seconds

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

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122212, upper bound: 0.0123045
time: 2.83 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122975, upper bound: 0.0123130
time: 2.52 seconds

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

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122212, upper bound: 0.0123572
time: 2.86 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122975, upper bound: 0.0123648
time: 2.64 seconds

## BFS NS instance: NS_A2_B1_A1_B1

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

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122924, upper bound: 0.0123249
time: 5.67 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123652, upper bound: 0.0123296
time: 2.63 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042693, 0.0030647, -0.0046388, 0.0028599, -0.0071292, 0.0077035
1: 0.9887339, 1.0054924, 0.9892505, 1.0062261, -0.0174921, 0.0162419
2: -0.0143153, 0.0032835, -0.0134166, 0.0036661, -0.0173543, 0.0161917
3: 0.0005192, 0.0060166, 0.0004158, 0.0058023, -0.0052832, 0.0056008
4: -0.0037652, 0.0097310, -0.0042515, 0.0090208, -0.0127859, 0.0139825
5: -0.0011647, 0.0112018, -0.0013330, 0.0106996, -0.0118644, 0.0125347
6: -0.0041817, 0.0039406, -0.0049329, 0.0032360, -0.0074177, 0.0088735
7: -0.0116780, -0.0016566, -0.0113164, -0.0013886, -0.0102894, 0.0096598
8: -0.0110738, 0.0170602, -0.0114483, 0.0158794, -0.0267970, 0.0283242
9: -0.0099894, 0.0061594, -0.0093337, 0.0063792, -0.0163685, 0.0154931

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122924, upper bound: 0.0123250
time: 3.02 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123652, upper bound: 0.0123299
time: 3.13 seconds

## BFS NS instance: NS_A2_B1_A2_B1

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

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123098, upper bound: 0.0122786
time: 2.92 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123858, upper bound: 0.0122847
time: 2.12 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0057187, 0.0037577, -0.0046388, 0.0028599, -0.0085786, 0.0083965
1: 0.9885953, 1.0083705, 0.9892505, 1.0062261, -0.0176308, 0.0191200
2: -0.0145566, 0.0047846, -0.0134166, 0.0036661, -0.0175227, 0.0176441
3: 0.0001134, 0.0060741, 0.0004158, 0.0058023, -0.0056889, 0.0056584
4: -0.0056728, 0.0099217, -0.0042515, 0.0090208, -0.0146936, 0.0141732
5: -0.0018248, 0.0113366, -0.0013330, 0.0106996, -0.0125245, 0.0126695
6: -0.0071285, 0.0041298, -0.0049329, 0.0032360, -0.0103644, 0.0090627
7: -0.0117751, -0.0006053, -0.0113164, -0.0013886, -0.0103865, 0.0107112
8: -0.0125432, 0.0173771, -0.0114483, 0.0158794, -0.0282543, 0.0286205
9: -0.0101654, 0.0070215, -0.0093337, 0.0063792, -0.0165446, 0.0163551

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123098, upper bound: 0.0122783
time: 2.78 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123858, upper bound: 0.0122855
time: 2.30 seconds

## BFS NS instance: NS_A2_B2_A1_B1

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

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122894, upper bound: 0.0123249
time: 2.73 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123639, upper bound: 0.0123300
time: 2.79 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042693, 0.0030647, -0.0057187, 0.0037577, -0.0080270, 0.0087834
1: 0.9887339, 1.0054924, 0.9885953, 1.0083705, -0.0196366, 0.0168971
2: -0.0143153, 0.0032835, -0.0145566, 0.0047846, -0.0185393, 0.0172800
3: 0.0005192, 0.0060166, 0.0001134, 0.0060741, -0.0055549, 0.0059032
4: -0.0037652, 0.0097310, -0.0056728, 0.0099217, -0.0136869, 0.0154039
5: -0.0011647, 0.0112018, -0.0018248, 0.0113366, -0.0125013, 0.0130266
6: -0.0041817, 0.0039406, -0.0071285, 0.0041298, -0.0083115, 0.0110691
7: -0.0116780, -0.0016566, -0.0117751, -0.0006053, -0.0110727, 0.0101184
8: -0.0110738, 0.0170602, -0.0125432, 0.0173771, -0.0282798, 0.0294349
9: -0.0099894, 0.0061594, -0.0101654, 0.0070215, -0.0170108, 0.0163248

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122894, upper bound: 0.0123244
time: 2.56 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123639, upper bound: 0.0123296
time: 2.51 seconds

## BFS NS instance: NS_A2_B2_A2_B1

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

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123071, upper bound: 0.0122783
time: 2.93 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123847, upper bound: 0.0122855
time: 2.23 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0057187, 0.0037577, -0.0057187, 0.0037577, -0.0094764, 0.0094764
1: 0.9885953, 1.0083705, 0.9885953, 1.0083705, -0.0197752, 0.0197752
2: -0.0145566, 0.0047846, -0.0145566, 0.0047846, -0.0187315, 0.0187315
3: 0.0001134, 0.0060741, 0.0001134, 0.0060741, -0.0059607, 0.0059607
4: -0.0056728, 0.0099217, -0.0056728, 0.0099217, -0.0155946, 0.0155946
5: -0.0018248, 0.0113366, -0.0018248, 0.0113366, -0.0131614, 0.0131614
6: -0.0071285, 0.0041298, -0.0071285, 0.0041298, -0.0112582, 0.0112582
7: -0.0117751, -0.0006053, -0.0117751, -0.0006053, -0.0111698, 0.0111698
8: -0.0125432, 0.0173771, -0.0125432, 0.0173771, -0.0297375, 0.0297375
9: -0.0101654, 0.0070215, -0.0101654, 0.0070215, -0.0171868, 0.0171868

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123068, upper bound: 0.0122781
time: 2.48 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123847, upper bound: 0.0122852
time: 2.43 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.57 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0122500, upper bound: 0.0123850
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0123457, upper bound: 0.0123926
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0122500, upper bound: 0.0124253
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0123457, upper bound: 0.0124303
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0122377, upper bound: 0.0123046
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0123297, upper bound: 0.0123138
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0122377, upper bound: 0.0123568
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0123297, upper bound: 0.0123648
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0122003, upper bound: 0.0123684
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0122857, upper bound: 0.0123743
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0122003, upper bound: 0.0124146
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0122857, upper bound: 0.0124203
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0122212, upper bound: 0.0123045
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0122975, upper bound: 0.0123130
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0122212, upper bound: 0.0123572
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0122975, upper bound: 0.0123648
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0122924, upper bound: 0.0123249
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0123652, upper bound: 0.0123296
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0122924, upper bound: 0.0123250
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0123652, upper bound: 0.0123299
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0123098, upper bound: 0.0122786
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0123858, upper bound: 0.0122847
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0123098, upper bound: 0.0122783
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0123858, upper bound: 0.0122855
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0122894, upper bound: 0.0123249
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0123639, upper bound: 0.0123300
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0122894, upper bound: 0.0123244
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0123639, upper bound: 0.0123296
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0123071, upper bound: 0.0122783
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0123847, upper bound: 0.0122855
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0123068, upper bound: 0.0122781
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.57
Output dim: 1, lower bound: -0.0123847, upper bound: 0.0122852

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

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121573, upper bound: 0.0122948
time: 2.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122380, upper bound: 0.0123205
time: 2.81 seconds

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

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123853, upper bound: 0.0123040
time: 2.50 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123853, upper bound: 0.0123040
time: 2.21 seconds

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

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121053, upper bound: 0.0123275
time: 3.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121861, upper bound: 0.0123622
time: 3.97 seconds

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

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123411, upper bound: 0.0123571
time: 3.17 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123411, upper bound: 0.0124311
time: 3.28 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0044253, 0.0026286, -0.0031722, 0.0027732, -0.0071984, 0.0058008
1: 0.9898338, 1.0058022, 0.9894690, 1.0033140, -0.0134802, 0.0163332
2: -0.0124021, 0.0034450, -0.0130364, 0.0021473, -0.0138960, 0.0158390
3: 0.0004755, 0.0055604, 0.0008263, 0.0057117, -0.0052361, 0.0047341
4: -0.0039704, 0.0082190, -0.0023212, 0.0087202, -0.0126907, 0.0105401
5: -0.0012358, 0.0101328, -0.0006650, 0.0104872, -0.0117229, 0.0107978
6: -0.0044988, 0.0024405, -0.0036461, 0.0029378, -0.0074366, 0.0060866
7: -0.0109083, -0.0015435, -0.0111635, -0.0024524, -0.0084559, 0.0096200
8: -0.0112319, 0.0145465, -0.0099615, 0.0153798, -0.0264194, 0.0243151
9: -0.0085935, 0.0062522, -0.0090562, 0.0055069, -0.0141004, 0.0153084

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121418, upper bound: 0.0122165
time: 2.94 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122231, upper bound: 0.0122394
time: 2.97 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0045276, 0.0028078, -0.0032417, 0.0028070, -0.0073346, 0.0060496
1: 0.9893818, 1.0060053, 0.9893840, 1.0034519, -0.0140702, 0.0166213
2: -0.0131884, 0.0035510, -0.0131847, 0.0022193, -0.0147193, 0.0160899
3: 0.0004469, 0.0057479, 0.0008069, 0.0057470, -0.0053002, 0.0049410
4: -0.0041051, 0.0088404, -0.0024127, 0.0088374, -0.0129426, 0.0112531
5: -0.0012824, 0.0105721, -0.0006967, 0.0105700, -0.0118524, 0.0112689
6: -0.0047069, 0.0030570, -0.0036976, 0.0030541, -0.0077610, 0.0067546
7: -0.0112246, -0.0014692, -0.0112231, -0.0024020, -0.0088226, 0.0097539
8: -0.0113357, 0.0155796, -0.0100320, 0.0155746, -0.0267236, 0.0254082
9: -0.0091672, 0.0063130, -0.0091644, 0.0055483, -0.0147154, 0.0154775

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123685, upper bound: 0.0122351
time: 2.17 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123685, upper bound: 0.0123135
time: 2.67 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0044253, 0.0026286, -0.0041942, 0.0030308, -0.0074561, 0.0068227
1: 0.9898338, 1.0058022, 0.9888196, 1.0053432, -0.0155094, 0.0169826
2: -0.0124021, 0.0034450, -0.0141664, 0.0032057, -0.0149864, 0.0169899
3: 0.0004755, 0.0055604, 0.0005402, 0.0059811, -0.0055056, 0.0050202
4: -0.0039704, 0.0082190, -0.0036662, 0.0096134, -0.0135838, 0.0118852
5: -0.0012358, 0.0101328, -0.0011305, 0.0111186, -0.0123543, 0.0112633
6: -0.0044988, 0.0024405, -0.0040384, 0.0038239, -0.0083227, 0.0064788
7: -0.0109083, -0.0015435, -0.0116181, -0.0017111, -0.0091971, 0.0100746
8: -0.0112319, 0.0145465, -0.0109976, 0.0168645, -0.0279085, 0.0253581
9: -0.0085935, 0.0062522, -0.0098807, 0.0061147, -0.0147082, 0.0161329

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0120896, upper bound: 0.0122622
time: 3.09 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121744, upper bound: 0.0122916
time: 6.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0045276, 0.0028078, -0.0042693, 0.0030647, -0.0075923, 0.0070772
1: 0.9893818, 1.0060053, 0.9887339, 1.0054924, -0.0161107, 0.0172713
2: -0.0131884, 0.0035510, -0.0143153, 0.0032835, -0.0159650, 0.0172395
3: 0.0004469, 0.0057479, 0.0005192, 0.0060166, -0.0055697, 0.0052287
4: -0.0041051, 0.0088404, -0.0037652, 0.0097310, -0.0138362, 0.0126056
5: -0.0012824, 0.0105721, -0.0011647, 0.0112018, -0.0124841, 0.0117369
6: -0.0047069, 0.0030570, -0.0041817, 0.0039406, -0.0086475, 0.0072387
7: -0.0112246, -0.0014692, -0.0116780, -0.0016566, -0.0095680, 0.0102087
8: -0.0113357, 0.0155796, -0.0110738, 0.0170602, -0.0282115, 0.0264952
9: -0.0091672, 0.0063130, -0.0099894, 0.0061594, -0.0153266, 0.0163024

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123253, upper bound: 0.0122917
time: 2.84 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123253, upper bound: 0.0123648
time: 2.45 seconds

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

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0120890, upper bound: 0.0122803
time: 2.85 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121692, upper bound: 0.0123047
time: 2.85 seconds

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

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123046, upper bound: 0.0122891
time: 2.73 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123046, upper bound: 0.0123749
time: 4.18 seconds

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

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0120559, upper bound: 0.0123224
time: 3.41 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121351, upper bound: 0.0123523
time: 3.34 seconds

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

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122787, upper bound: 0.0123465
time: 3.29 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122787, upper bound: 0.0124203
time: 2.72 seconds

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

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121000, upper bound: 0.0122158
time: 2.79 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121851, upper bound: 0.0122389
time: 2.28 seconds

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

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123186, upper bound: 0.0122351
time: 2.61 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123186, upper bound: 0.0123135
time: 2.62 seconds

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

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0120660, upper bound: 0.0122630
time: 2.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121561, upper bound: 0.0122918
time: 2.93 seconds

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

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122912, upper bound: 0.0122920
time: 2.91 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122912, upper bound: 0.0123643
time: 2.73 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0040592, 0.0028335, -0.0031722, 0.0027732, -0.0068324, 0.0060058
1: 0.9893169, 1.0050751, 0.9894690, 1.0033140, -0.0139971, 0.0156061
2: -0.0133013, 0.0030659, -0.0130364, 0.0021473, -0.0147610, 0.0154330
3: 0.0005780, 0.0057748, 0.0008263, 0.0057117, -0.0051336, 0.0049485
4: -0.0034886, 0.0089296, -0.0023212, 0.0087202, -0.0122088, 0.0112507
5: -0.0010690, 0.0106352, -0.0006650, 0.0104872, -0.0115562, 0.0113002
6: -0.0037546, 0.0031455, -0.0036461, 0.0029378, -0.0066924, 0.0067916
7: -0.0112700, -0.0018090, -0.0111635, -0.0024524, -0.0088176, 0.0093544
8: -0.0108608, 0.0157278, -0.0099615, 0.0153798, -0.0260452, 0.0254887
9: -0.0092495, 0.0060345, -0.0090562, 0.0055069, -0.0147564, 0.0150907

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121969, upper bound: 0.0122458
time: 2.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122936, upper bound: 0.0122779
time: 2.97 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0041496, 0.0030145, -0.0032417, 0.0028070, -0.0069565, 0.0062562
1: 0.9888607, 1.0052546, 0.9893840, 1.0034519, -0.0145912, 0.0158706
2: -0.0140949, 0.0031595, -0.0131847, 0.0022193, -0.0156008, 0.0156713
3: 0.0005527, 0.0059641, 0.0008069, 0.0057470, -0.0051943, 0.0051572
4: -0.0036075, 0.0095569, -0.0024127, 0.0088374, -0.0124450, 0.0119696
5: -0.0011102, 0.0110787, -0.0006967, 0.0105700, -0.0116802, 0.0117754
6: -0.0040136, 0.0037678, -0.0036976, 0.0030541, -0.0070676, 0.0074654
7: -0.0115893, -0.0017435, -0.0112231, -0.0024020, -0.0091873, 0.0094796
8: -0.0109523, 0.0167707, -0.0100320, 0.0155746, -0.0263330, 0.0265976
9: -0.0098286, 0.0060882, -0.0091644, 0.0055483, -0.0153769, 0.0152526

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124253, upper bound: 0.0122497
time: 2.90 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124253, upper bound: 0.0123457
time: 2.54 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0040592, 0.0028335, -0.0045712, 0.0028268, -0.0068860, 0.0074047
1: 0.9893169, 1.0050751, 0.9893340, 1.0060917, -0.0167748, 0.0157411
2: -0.0133013, 0.0030659, -0.0132715, 0.0035961, -0.0162689, 0.0157136
3: 0.0005780, 0.0057748, 0.0004347, 0.0057677, -0.0051897, 0.0053401
4: -0.0034886, 0.0089296, -0.0041624, 0.0089061, -0.0123947, 0.0130920
5: -0.0010690, 0.0106352, -0.0013022, 0.0106186, -0.0116876, 0.0119374
6: -0.0037546, 0.0031455, -0.0047954, 0.0031222, -0.0068768, 0.0079409
7: -0.0112700, -0.0018090, -0.0112581, -0.0014377, -0.0098323, 0.0094491
8: -0.0108608, 0.0157278, -0.0113798, 0.0156888, -0.0263615, 0.0269212
9: -0.0092495, 0.0060345, -0.0092278, 0.0063389, -0.0155884, 0.0152623

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121384, upper bound: 0.0122333
time: 2.88 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122274, upper bound: 0.0122630
time: 2.60 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0041496, 0.0030145, -0.0046388, 0.0028599, -0.0070094, 0.0076533
1: 0.9888607, 1.0052546, 0.9892505, 1.0062261, -0.0173653, 0.0160041
2: -0.0140949, 0.0031595, -0.0134166, 0.0036661, -0.0171103, 0.0159478
3: 0.0005527, 0.0059641, 0.0004158, 0.0058023, -0.0052496, 0.0055483
4: -0.0036075, 0.0095569, -0.0042515, 0.0090208, -0.0126283, 0.0138084
5: -0.0011102, 0.0110787, -0.0013330, 0.0106996, -0.0118098, 0.0124116
6: -0.0040136, 0.0037678, -0.0049329, 0.0032360, -0.0072495, 0.0087007
7: -0.0115893, -0.0017435, -0.0113164, -0.0013886, -0.0102007, 0.0095730
8: -0.0109523, 0.0167707, -0.0114483, 0.0158794, -0.0266440, 0.0280283
9: -0.0098286, 0.0060882, -0.0093337, 0.0063792, -0.0162078, 0.0154219

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123572, upper bound: 0.0122375
time: 3.03 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123569, upper bound: 0.0123300
time: 2.82 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0055040, 0.0034767, -0.0031722, 0.0027732, -0.0082772, 0.0066489
1: 0.9891715, 1.0079441, 0.9894690, 1.0033140, -0.0141425, 0.0184751
2: -0.0135542, 0.0045622, -0.0130364, 0.0021473, -0.0150882, 0.0169996
3: 0.0001735, 0.0058351, 0.0008263, 0.0057117, -0.0055381, 0.0050088
4: -0.0053903, 0.0091295, -0.0023212, 0.0087202, -0.0141105, 0.0114507
5: -0.0017271, 0.0107766, -0.0006650, 0.0104872, -0.0122142, 0.0114416
6: -0.0066920, 0.0033438, -0.0036461, 0.0029378, -0.0096298, 0.0069900
7: -0.0113718, -0.0007610, -0.0111635, -0.0024524, -0.0089194, 0.0104025
8: -0.0123255, 0.0160602, -0.0099615, 0.0153798, -0.0275246, 0.0258354
9: -0.0094341, 0.0068938, -0.0090562, 0.0055069, -0.0149410, 0.0159500

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121868, upper bound: 0.0121906
time: 2.80 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122825, upper bound: 0.0122143
time: 2.64 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0055719, 0.0035655, -0.0032417, 0.0028070, -0.0083789, 0.0068073
1: 0.9887300, 1.0080789, 0.9893840, 1.0034519, -0.0147220, 0.0186949
2: -0.0143222, 0.0046325, -0.0131847, 0.0022193, -0.0159004, 0.0172097
3: 0.0001545, 0.0060182, 0.0008069, 0.0057470, -0.0055925, 0.0052114
4: -0.0054797, 0.0097365, -0.0024127, 0.0088374, -0.0143171, 0.0121491
5: -0.0017580, 0.0112056, -0.0006967, 0.0105700, -0.0123280, 0.0119023
6: -0.0068301, 0.0039460, -0.0036976, 0.0030541, -0.0098841, 0.0076436
7: -0.0116807, -0.0007117, -0.0112231, -0.0024020, -0.0092787, 0.0105114
8: -0.0123944, 0.0170692, -0.0100320, 0.0155746, -0.0277878, 0.0269078
9: -0.0099944, 0.0069342, -0.0091644, 0.0055483, -0.0155426, 0.0160986

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124150, upper bound: 0.0122001
time: 2.40 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124150, upper bound: 0.0122853
time: 2.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0055040, 0.0034767, -0.0045712, 0.0028268, -0.0083308, 0.0080478
1: 0.9891715, 1.0079441, 0.9893340, 1.0060917, -0.0169202, 0.0186101
2: -0.0135542, 0.0045622, -0.0132715, 0.0035961, -0.0164478, 0.0171412
3: 0.0001735, 0.0058351, 0.0004347, 0.0057677, -0.0055942, 0.0054005
4: -0.0053903, 0.0091295, -0.0041624, 0.0089061, -0.0142964, 0.0132920
5: -0.0017271, 0.0107766, -0.0013022, 0.0106186, -0.0123456, 0.0120787
6: -0.0066920, 0.0033438, -0.0047954, 0.0031222, -0.0098142, 0.0081392
7: -0.0113718, -0.0007610, -0.0112581, -0.0014377, -0.0099341, 0.0104971
8: -0.0123255, 0.0160602, -0.0113798, 0.0156888, -0.0278085, 0.0272357
9: -0.0094341, 0.0068938, -0.0092278, 0.0063389, -0.0157730, 0.0161216

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121554, upper bound: 0.0121909
time: 2.35 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122449, upper bound: 0.0122139
time: 2.69 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0055719, 0.0035655, -0.0046388, 0.0028599, -0.0084318, 0.0082043
1: 0.9887300, 1.0080789, 0.9892505, 1.0062261, -0.0174961, 0.0188284
2: -0.0143222, 0.0046325, -0.0134166, 0.0036661, -0.0172653, 0.0173485
3: 0.0001545, 0.0060182, 0.0004158, 0.0058023, -0.0056478, 0.0056025
4: -0.0054797, 0.0097365, -0.0042515, 0.0090208, -0.0145005, 0.0139879
5: -0.0017580, 0.0112056, -0.0013330, 0.0106996, -0.0124576, 0.0125386
6: -0.0068301, 0.0039460, -0.0049329, 0.0032360, -0.0100660, 0.0088789
7: -0.0116807, -0.0007117, -0.0113164, -0.0013886, -0.0102921, 0.0106047
8: -0.0123944, 0.0170692, -0.0114483, 0.0158794, -0.0280669, 0.0283070
9: -0.0099944, 0.0069342, -0.0093337, 0.0063792, -0.0163735, 0.0162678

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123781, upper bound: 0.0121998
time: 3.06 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123781, upper bound: 0.0122852
time: 2.54 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

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

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121844, upper bound: 0.0122467
time: 3.30 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122839, upper bound: 0.0122779
time: 3.59 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

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

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124166, upper bound: 0.0122495
time: 2.55 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124166, upper bound: 0.0123459
time: 2.65 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0040592, 0.0028335, -0.0056266, 0.0036371, -0.0076963, 0.0084601
1: 0.9893169, 1.0050751, 0.9886786, 1.0081874, -0.0188705, 0.0163965
2: -0.0133013, 0.0030659, -0.0144112, 0.0046891, -0.0172866, 0.0167842
3: 0.0005780, 0.0057748, 0.0001392, 0.0060395, -0.0054615, 0.0056356
4: -0.0034886, 0.0089296, -0.0055516, 0.0098069, -0.0132955, 0.0144812
5: -0.0010690, 0.0106352, -0.0017829, 0.0112554, -0.0123244, 0.0124181
6: -0.0037546, 0.0031455, -0.0069412, 0.0040158, -0.0077704, 0.0100867
7: -0.0112700, -0.0018090, -0.0117166, -0.0006721, -0.0105979, 0.0099076
8: -0.0108608, 0.0157278, -0.0124498, 0.0171862, -0.0278398, 0.0279697
9: -0.0092495, 0.0060345, -0.0100594, 0.0069667, -0.0162162, 0.0160938

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121350, upper bound: 0.0122341
time: 2.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122244, upper bound: 0.0122627
time: 2.51 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0041496, 0.0030145, -0.0057187, 0.0037577, -0.0079073, 0.0087332
1: 0.9888607, 1.0052546, 0.9885953, 1.0083705, -0.0195098, 0.0166593
2: -0.0140949, 0.0031595, -0.0145566, 0.0047846, -0.0183217, 0.0170217
3: 0.0005527, 0.0059641, 0.0001134, 0.0060741, -0.0055214, 0.0058506
4: -0.0036075, 0.0095569, -0.0056728, 0.0099217, -0.0135292, 0.0152298
5: -0.0011102, 0.0110787, -0.0018248, 0.0113366, -0.0124467, 0.0129035
6: -0.0040136, 0.0037678, -0.0071285, 0.0041298, -0.0081433, 0.0108963
7: -0.0115893, -0.0017435, -0.0117751, -0.0006053, -0.0109841, 0.0100316
8: -0.0109523, 0.0167707, -0.0125432, 0.0173771, -0.0281231, 0.0291458
9: -0.0098286, 0.0060882, -0.0101654, 0.0070215, -0.0168501, 0.0162536

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123558, upper bound: 0.0122378
time: 2.56 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123558, upper bound: 0.0123294
time: 2.59 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0055040, 0.0034767, -0.0041942, 0.0030308, -0.0085348, 0.0076708
1: 0.9891715, 1.0079441, 0.9888196, 1.0053432, -0.0161717, 0.0191245
2: -0.0135542, 0.0045622, -0.0141664, 0.0032057, -0.0160617, 0.0180400
3: 0.0001735, 0.0058351, 0.0005402, 0.0059811, -0.0058076, 0.0052949
4: -0.0053903, 0.0091295, -0.0036662, 0.0096134, -0.0150036, 0.0127958
5: -0.0017271, 0.0107766, -0.0011305, 0.0111186, -0.0128457, 0.0119070
6: -0.0066920, 0.0033438, -0.0040384, 0.0038239, -0.0105159, 0.0073822
7: -0.0113718, -0.0007610, -0.0116181, -0.0017111, -0.0096607, 0.0108571
8: -0.0123255, 0.0160602, -0.0109976, 0.0168645, -0.0289869, 0.0268516
9: -0.0094341, 0.0068938, -0.0098807, 0.0061147, -0.0155488, 0.0167745

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121772, upper bound: 0.0121906
time: 2.77 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122751, upper bound: 0.0122140
time: 2.41 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0055719, 0.0035655, -0.0042693, 0.0030647, -0.0086367, 0.0078349
1: 0.9887300, 1.0080789, 0.9887339, 1.0054924, -0.0167625, 0.0193450
2: -0.0143222, 0.0046325, -0.0143153, 0.0032835, -0.0170441, 0.0182492
3: 0.0001545, 0.0060182, 0.0005192, 0.0060166, -0.0058621, 0.0054991
4: -0.0054797, 0.0097365, -0.0037652, 0.0097310, -0.0152107, 0.0135016
5: -0.0017580, 0.0112056, -0.0011647, 0.0112018, -0.0129598, 0.0123703
6: -0.0068301, 0.0039460, -0.0041817, 0.0039406, -0.0107707, 0.0081277
7: -0.0116807, -0.0007117, -0.0116780, -0.0016566, -0.0100241, 0.0109662
8: -0.0123944, 0.0170692, -0.0110738, 0.0170602, -0.0292493, 0.0279718
9: -0.0099944, 0.0069342, -0.0099894, 0.0061594, -0.0161538, 0.0169235

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124091, upper bound: 0.0122003
time: 2.67 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0124091, upper bound: 0.0122853
time: 2.37 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0055040, 0.0034767, -0.0056266, 0.0036371, -0.0091411, 0.0091032
1: 0.9891715, 1.0079441, 0.9886786, 1.0081874, -0.0190159, 0.0192655
2: -0.0135542, 0.0045622, -0.0144112, 0.0046891, -0.0174678, 0.0182102
3: 0.0001735, 0.0058351, 0.0001392, 0.0060395, -0.0058660, 0.0056959
4: -0.0053903, 0.0091295, -0.0055516, 0.0098069, -0.0151971, 0.0146811
5: -0.0017271, 0.0107766, -0.0017829, 0.0112554, -0.0129825, 0.0125594
6: -0.0066920, 0.0033438, -0.0069412, 0.0040158, -0.0107078, 0.0102850
7: -0.0113718, -0.0007610, -0.0117166, -0.0006721, -0.0106997, 0.0109556
8: -0.0123255, 0.0160602, -0.0124498, 0.0171862, -0.0292882, 0.0282852
9: -0.0094341, 0.0068938, -0.0100594, 0.0069667, -0.0164007, 0.0169531

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121519, upper bound: 0.0121905
time: 2.71 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122422, upper bound: 0.0122138
time: 2.41 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0055719, 0.0035655, -0.0057187, 0.0037577, -0.0093297, 0.0092843
1: 0.9887300, 1.0080789, 0.9885953, 1.0083705, -0.0196406, 0.0194836
2: -0.0143222, 0.0046325, -0.0145566, 0.0047846, -0.0184970, 0.0184192
3: 0.0001545, 0.0060182, 0.0001134, 0.0060741, -0.0059196, 0.0059048
4: -0.0054797, 0.0097365, -0.0056728, 0.0099217, -0.0154014, 0.0154093
5: -0.0017580, 0.0112056, -0.0018248, 0.0113366, -0.0130945, 0.0130304
6: -0.0068301, 0.0039460, -0.0071285, 0.0041298, -0.0109598, 0.0110745
7: -0.0116807, -0.0007117, -0.0117751, -0.0006053, -0.0110755, 0.0110633
8: -0.0123944, 0.0170692, -0.0125432, 0.0173771, -0.0295469, 0.0294296
9: -0.0099944, 0.0069342, -0.0101654, 0.0070215, -0.0170158, 0.0170995

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123770, upper bound: 0.0122000
time: 2.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123770, upper bound: 0.0122856
time: 2.75 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 7.06 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0121573, upper bound: 0.0122948
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0122380, upper bound: 0.0123205
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123853, upper bound: 0.0123040
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123853, upper bound: 0.0123040
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0121053, upper bound: 0.0123275
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0121861, upper bound: 0.0123622
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123411, upper bound: 0.0123571
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123411, upper bound: 0.0124311
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0121418, upper bound: 0.0122165
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0122231, upper bound: 0.0122394
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123685, upper bound: 0.0122351
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123685, upper bound: 0.0123135
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0120896, upper bound: 0.0122622
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0121744, upper bound: 0.0122916
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123253, upper bound: 0.0122917
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123253, upper bound: 0.0123648
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0120890, upper bound: 0.0122803
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0121692, upper bound: 0.0123047
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123046, upper bound: 0.0122891
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123046, upper bound: 0.0123749
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0120559, upper bound: 0.0123224
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0121351, upper bound: 0.0123523
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0122787, upper bound: 0.0123465
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0122787, upper bound: 0.0124203
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0121000, upper bound: 0.0122158
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0121851, upper bound: 0.0122389
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123186, upper bound: 0.0122351
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123186, upper bound: 0.0123135
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0120660, upper bound: 0.0122630
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0121561, upper bound: 0.0122918
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0122912, upper bound: 0.0122920
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0122912, upper bound: 0.0123643
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0121969, upper bound: 0.0122458
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0122936, upper bound: 0.0122779
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0124253, upper bound: 0.0122497
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0124253, upper bound: 0.0123457
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0121384, upper bound: 0.0122333
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0122274, upper bound: 0.0122630
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123572, upper bound: 0.0122375
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123569, upper bound: 0.0123300
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0121868, upper bound: 0.0121906
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0122825, upper bound: 0.0122143
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0124150, upper bound: 0.0122001
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0124150, upper bound: 0.0122853
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0121554, upper bound: 0.0121909
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0122449, upper bound: 0.0122139
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123781, upper bound: 0.0121998
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123781, upper bound: 0.0122852
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0121844, upper bound: 0.0122467
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0122839, upper bound: 0.0122779
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0124166, upper bound: 0.0122495
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0124166, upper bound: 0.0123459
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0121350, upper bound: 0.0122341
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0122244, upper bound: 0.0122627
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123558, upper bound: 0.0122378
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123558, upper bound: 0.0123294
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0121772, upper bound: 0.0121906
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0122751, upper bound: 0.0122140
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0124091, upper bound: 0.0122003
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0124091, upper bound: 0.0122853
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0121519, upper bound: 0.0121905
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0122422, upper bound: 0.0122138
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123770, upper bound: 0.0122000
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.06
Output dim: 1, lower bound: -0.0123770, upper bound: 0.0122856

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0029290, 0.0025710, -0.0025065, 0.0027774, -0.0057065, 0.0050775
1: 0.9899789, 1.0028311, 0.9894583, 1.0019923, -0.0120134, 0.0133728
2: -0.0121497, 0.0018954, -0.0130551, 0.0014579, -0.0128987, 0.0142376
3: 0.0008944, 0.0055003, 0.0010127, 0.0057161, -0.0048217, 0.0044876
4: -0.0020011, 0.0080195, -0.0022204, 0.0087351, -0.0107362, 0.0102399
5: -0.0005543, 0.0099918, -0.0003618, 0.0104977, -0.0110520, 0.0103536
6: -0.0033384, 0.0022426, -0.0036527, 0.0029525, -0.0062909, 0.0058952
7: -0.0108067, -0.0026288, -0.0111710, -0.0029353, -0.0078714, 0.0085422
8: -0.0097149, 0.0142149, -0.0092866, 0.0154045, -0.0249142, 0.0232955
9: -0.0084093, 0.0053623, -0.0090699, 0.0051110, -0.0135203, 0.0144322

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121557, upper bound: 0.0122232
time: 2.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121557, upper bound: 0.0122943
time: 2.68 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0030213, 0.0025732, -0.0028176, 0.0027667, -0.0057879, 0.0053907
1: 0.9899734, 1.0030142, 0.9894856, 1.0026100, -0.0126365, 0.0135286
2: -0.0121592, 0.0019909, -0.0130079, 0.0017800, -0.0132157, 0.0142872
3: 0.0008686, 0.0055025, 0.0009256, 0.0057049, -0.0048363, 0.0045769
4: -0.0021225, 0.0080270, -0.0021732, 0.0086977, -0.0108202, 0.0102002
5: -0.0005963, 0.0099971, -0.0005035, 0.0104713, -0.0110675, 0.0105006
6: -0.0033417, 0.0022500, -0.0036362, 0.0029154, -0.0062571, 0.0058862
7: -0.0108105, -0.0025619, -0.0111520, -0.0027097, -0.0081009, 0.0085900
8: -0.0098084, 0.0142273, -0.0096019, 0.0153424, -0.0249467, 0.0236222
9: -0.0084162, 0.0054171, -0.0090354, 0.0052960, -0.0137122, 0.0144526

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122025, upper bound: 0.0122236
time: 3.58 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122029, upper bound: 0.0123202
time: 2.46 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0031307, 0.0027583, -0.0030213, 0.0025732, -0.0057038, 0.0057796
1: 0.9895065, 1.0032316, 0.9899734, 1.0030142, -0.0135077, 0.0132582
2: -0.0129715, 0.0021043, -0.0121592, 0.0019909, -0.0142551, 0.0135521
3: 0.0008379, 0.0056962, 0.0008686, 0.0055025, -0.0046646, 0.0048276
4: -0.0022665, 0.0086689, -0.0021225, 0.0080270, -0.0102935, 0.0107914
5: -0.0006461, 0.0104509, -0.0005963, 0.0099971, -0.0106432, 0.0110472
6: -0.0036236, 0.0028869, -0.0033417, 0.0022500, -0.0058736, 0.0062285
7: -0.0111373, -0.0024826, -0.0108105, -0.0025619, -0.0085754, 0.0083280
8: -0.0099194, 0.0152945, -0.0098084, 0.0142273, -0.0239424, 0.0249003
9: -0.0090089, 0.0054822, -0.0084162, 0.0054171, -0.0144260, 0.0138984

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122949, upper bound: 0.0121570
time: 2.62 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123205, upper bound: 0.0122378
time: 2.47 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0031307, 0.0027583, -0.0031307, 0.0027583, -0.0058890, 0.0058890
1: 0.9895065, 1.0032316, 0.9895065, 1.0032316, -0.0137252, 0.0137252
2: -0.0129715, 0.0021043, -0.0129715, 0.0021043, -0.0143420, 0.0143420
3: 0.0008379, 0.0056962, 0.0008379, 0.0056962, -0.0048582, 0.0048582
4: -0.0022665, 0.0086689, -0.0022665, 0.0086689, -0.0109354, 0.0109354
5: -0.0006461, 0.0104509, -0.0006461, 0.0104509, -0.0110970, 0.0110970
6: -0.0036236, 0.0028869, -0.0036236, 0.0028869, -0.0065105, 0.0065105
7: -0.0111373, -0.0024826, -0.0111373, -0.0024826, -0.0086548, 0.0086548
8: -0.0099194, 0.0152945, -0.0099194, 0.0152945, -0.0250059, 0.0250059
9: -0.0090089, 0.0054822, -0.0090089, 0.0054822, -0.0144911, 0.0144911

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122953, upper bound: 0.0121959
time: 2.97 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123208, upper bound: 0.0122721
time: 2.72 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0029290, 0.0025710, -0.0035392, 0.0030378, -0.0059668, 0.0061102
1: 0.9899789, 1.0028311, 0.9888020, 1.0040426, -0.0140637, 0.0140291
2: -0.0121497, 0.0018954, -0.0141970, 0.0025274, -0.0140040, 0.0153983
3: 0.0008944, 0.0055003, 0.0007236, 0.0059884, -0.0050940, 0.0047767
4: -0.0020011, 0.0080195, -0.0033596, 0.0096375, -0.0116386, 0.0113791
5: -0.0005543, 0.0099918, -0.0008322, 0.0111356, -0.0116899, 0.0108240
6: -0.0033384, 0.0022426, -0.0040490, 0.0038478, -0.0071862, 0.0062915
7: -0.0108067, -0.0026288, -0.0116304, -0.0021862, -0.0086205, 0.0090016
8: -0.0097149, 0.0142149, -0.0103336, 0.0169047, -0.0264181, 0.0243522
9: -0.0084093, 0.0053623, -0.0099031, 0.0057252, -0.0141345, 0.0152653

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121045, upper bound: 0.0122615
time: 2.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121045, upper bound: 0.0123281
time: 2.57 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0030213, 0.0025732, -0.0038372, 0.0030242, -0.0060454, 0.0064104
1: 0.9899734, 1.0030142, 0.9888362, 1.0046344, -0.0146610, 0.0141780
2: -0.0121592, 0.0019909, -0.0141373, 0.0028360, -0.0143068, 0.0154372
3: 0.0008686, 0.0055025, 0.0006401, 0.0059742, -0.0051056, 0.0048624
4: -0.0021225, 0.0080270, -0.0033001, 0.0095904, -0.0117128, 0.0113270
5: -0.0005963, 0.0099971, -0.0009679, 0.0111023, -0.0116986, 0.0109650
6: -0.0033417, 0.0022500, -0.0040283, 0.0038010, -0.0071427, 0.0062782
7: -0.0108105, -0.0025619, -0.0116064, -0.0019701, -0.0088405, 0.0090445
8: -0.0098084, 0.0142273, -0.0106357, 0.0168263, -0.0264349, 0.0246623
9: -0.0084162, 0.0054171, -0.0098595, 0.0059024, -0.0143186, 0.0152766

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121511, upper bound: 0.0122615
time: 2.64 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121511, upper bound: 0.0123622
time: 3.23 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0031307, 0.0027583, -0.0040592, 0.0028335, -0.0059642, 0.0068176
1: 0.9895065, 1.0032316, 0.9893169, 1.0050751, -0.0155686, 0.0139147
2: -0.0129715, 0.0021043, -0.0133013, 0.0030659, -0.0153715, 0.0147187
3: 0.0008379, 0.0056962, 0.0005780, 0.0057748, -0.0049369, 0.0051182
4: -0.0022665, 0.0086689, -0.0034886, 0.0089296, -0.0111961, 0.0121575
5: -0.0006461, 0.0104509, -0.0010690, 0.0106352, -0.0112813, 0.0115199
6: -0.0036236, 0.0028869, -0.0037546, 0.0031455, -0.0067691, 0.0066415
7: -0.0111373, -0.0024826, -0.0112700, -0.0018090, -0.0093283, 0.0087875
8: -0.0099194, 0.0152945, -0.0108608, 0.0157278, -0.0254467, 0.0259614
9: -0.0090089, 0.0054822, -0.0092495, 0.0060345, -0.0150433, 0.0147317

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122465, upper bound: 0.0121965
time: 3.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122784, upper bound: 0.0122938
time: 2.91 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0031307, 0.0027583, -0.0041496, 0.0030145, -0.0061452, 0.0069079
1: 0.9895065, 1.0032316, 0.9888607, 1.0052546, -0.0157481, 0.0143709
2: -0.0129715, 0.0021043, -0.0140949, 0.0031595, -0.0154308, 0.0154869
3: 0.0008379, 0.0056962, 0.0005527, 0.0059641, -0.0051261, 0.0051435
4: -0.0022665, 0.0086689, -0.0036075, 0.0095569, -0.0118234, 0.0122765
5: -0.0006461, 0.0104509, -0.0011102, 0.0110787, -0.0117248, 0.0115611
6: -0.0036236, 0.0028869, -0.0040136, 0.0037678, -0.0073914, 0.0069004
7: -0.0111373, -0.0024826, -0.0115893, -0.0017435, -0.0093938, 0.0091068
8: -0.0099194, 0.0152945, -0.0109523, 0.0167707, -0.0264853, 0.0260458
9: -0.0090089, 0.0054822, -0.0098286, 0.0060882, -0.0150971, 0.0153108

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122465, upper bound: 0.0122347
time: 2.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122784, upper bound: 0.0123162
time: 2.43 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0043311, 0.0026264, -0.0025065, 0.0027774, -0.0071085, 0.0051329
1: 0.9898393, 1.0056150, 0.9894583, 1.0019923, -0.0121531, 0.0161567
2: -0.0123926, 0.0033474, -0.0130551, 0.0014579, -0.0131999, 0.0157584
3: 0.0005019, 0.0055582, 0.0010127, 0.0057161, -0.0052142, 0.0045455
4: -0.0038464, 0.0082115, -0.0022204, 0.0087351, -0.0125815, 0.0104319
5: -0.0011928, 0.0101275, -0.0003618, 0.0104977, -0.0116905, 0.0104894
6: -0.0043073, 0.0024330, -0.0036527, 0.0029525, -0.0072598, 0.0060857
7: -0.0109045, -0.0016118, -0.0111710, -0.0029353, -0.0079692, 0.0095592
8: -0.0111364, 0.0145340, -0.0092866, 0.0154045, -0.0263471, 0.0236263
9: -0.0085866, 0.0061961, -0.0090699, 0.0051110, -0.0136976, 0.0152661

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121409, upper bound: 0.0121573
time: 3.03 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121409, upper bound: 0.0122164
time: 2.54 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0044253, 0.0026286, -0.0028176, 0.0027667, -0.0071919, 0.0054461
1: 0.9898338, 1.0058022, 0.9894856, 1.0026100, -0.0127762, 0.0163165
2: -0.0124021, 0.0034450, -0.0130079, 0.0017800, -0.0135053, 0.0158096
3: 0.0004755, 0.0055604, 0.0009256, 0.0057049, -0.0052294, 0.0046348
4: -0.0039704, 0.0082190, -0.0021732, 0.0086977, -0.0126682, 0.0103922
5: -0.0012358, 0.0101328, -0.0005035, 0.0104713, -0.0117070, 0.0106363
6: -0.0044988, 0.0024405, -0.0036362, 0.0029154, -0.0074142, 0.0060767
7: -0.0109083, -0.0015435, -0.0111520, -0.0027097, -0.0081986, 0.0096085
8: -0.0112319, 0.0145465, -0.0096019, 0.0153424, -0.0263820, 0.0239505
9: -0.0085935, 0.0062522, -0.0090354, 0.0052960, -0.0138894, 0.0152876

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121883, upper bound: 0.0121576
time: 3.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121883, upper bound: 0.0122391
time: 2.57 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0045276, 0.0028078, -0.0030213, 0.0025732, -0.0071008, 0.0058291
1: 0.9893818, 1.0060053, 0.9899734, 1.0030142, -0.0136324, 0.0160319
2: -0.0131884, 0.0035510, -0.0121592, 0.0019909, -0.0145165, 0.0150577
3: 0.0004469, 0.0057479, 0.0008686, 0.0055025, -0.0050556, 0.0048793
4: -0.0041051, 0.0088404, -0.0021225, 0.0080270, -0.0121321, 0.0109629
5: -0.0012824, 0.0105721, -0.0005963, 0.0099971, -0.0112794, 0.0111684
6: -0.0047069, 0.0030570, -0.0033417, 0.0022500, -0.0069569, 0.0063987
7: -0.0112246, -0.0014692, -0.0108105, -0.0025619, -0.0086627, 0.0093413
8: -0.0113357, 0.0155796, -0.0098084, 0.0142273, -0.0253728, 0.0251893
9: -0.0091672, 0.0063130, -0.0084162, 0.0054171, -0.0145843, 0.0147293

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122809, upper bound: 0.0120885
time: 2.74 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123048, upper bound: 0.0121695
time: 2.62 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0045276, 0.0028078, -0.0031307, 0.0027583, -0.0072860, 0.0059385
1: 0.9893818, 1.0060053, 0.9895065, 1.0032316, -0.0138499, 0.0164988
2: -0.0131884, 0.0035510, -0.0129715, 0.0021043, -0.0146054, 0.0158502
3: 0.0004469, 0.0057479, 0.0008379, 0.0056962, -0.0052493, 0.0049100
4: -0.0041051, 0.0088404, -0.0022665, 0.0086689, -0.0127741, 0.0111069
5: -0.0012824, 0.0105721, -0.0006461, 0.0104509, -0.0117333, 0.0112183
6: -0.0047069, 0.0030570, -0.0036236, 0.0028869, -0.0075938, 0.0066806
7: -0.0112246, -0.0014692, -0.0111373, -0.0024826, -0.0087421, 0.0096681
8: -0.0113357, 0.0155796, -0.0099194, 0.0152945, -0.0264362, 0.0252959
9: -0.0091672, 0.0063130, -0.0090089, 0.0054822, -0.0146494, 0.0153219

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122809, upper bound: 0.0121310
time: 3.16 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0123048, upper bound: 0.0121982
time: 2.67 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0043311, 0.0026264, -0.0035392, 0.0030378, -0.0073688, 0.0061656
1: 0.9898393, 1.0056150, 0.9888020, 1.0040426, -0.0142034, 0.0168130
2: -0.0123926, 0.0033474, -0.0141970, 0.0025274, -0.0143052, 0.0169191
3: 0.0005019, 0.0055582, 0.0007236, 0.0059884, -0.0054865, 0.0048346
4: -0.0038464, 0.0082115, -0.0033596, 0.0096375, -0.0134839, 0.0115711
5: -0.0011928, 0.0101275, -0.0008322, 0.0111356, -0.0123285, 0.0109597
6: -0.0043073, 0.0024330, -0.0040490, 0.0038478, -0.0081551, 0.0064820
7: -0.0109045, -0.0016118, -0.0116304, -0.0021862, -0.0087183, 0.0100186
8: -0.0111364, 0.0145340, -0.0103336, 0.0169047, -0.0278511, 0.0246829
9: -0.0085866, 0.0061961, -0.0099031, 0.0057252, -0.0143117, 0.0160992

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0120891, upper bound: 0.0122060
time: 2.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0120891, upper bound: 0.0122628
time: 4.30 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0044253, 0.0026286, -0.0038372, 0.0030242, -0.0074495, 0.0064658
1: 0.9898338, 1.0058022, 0.9888362, 1.0046344, -0.0148006, 0.0169659
2: -0.0124021, 0.0034450, -0.0141373, 0.0028360, -0.0145964, 0.0169596
3: 0.0004755, 0.0055604, 0.0006401, 0.0059742, -0.0054987, 0.0049203
4: -0.0039704, 0.0082190, -0.0033001, 0.0095904, -0.0135608, 0.0115190
5: -0.0012358, 0.0101328, -0.0009679, 0.0111023, -0.0123381, 0.0111007
6: -0.0044988, 0.0024405, -0.0040283, 0.0038010, -0.0082999, 0.0064687
7: -0.0109083, -0.0015435, -0.0116064, -0.0019701, -0.0089382, 0.0100629
8: -0.0112319, 0.0145465, -0.0106357, 0.0168263, -0.0278702, 0.0249907
9: -0.0085935, 0.0062522, -0.0098595, 0.0059024, -0.0144959, 0.0161117

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121410, upper bound: 0.0122066
time: 2.41 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121410, upper bound: 0.0122922
time: 2.44 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0045276, 0.0028078, -0.0040592, 0.0028335, -0.0073612, 0.0068671
1: 0.9893818, 1.0060053, 0.9893169, 1.0050751, -0.0156933, 0.0166883
2: -0.0131884, 0.0035510, -0.0133013, 0.0030659, -0.0156328, 0.0162243
3: 0.0004469, 0.0057479, 0.0005780, 0.0057748, -0.0053279, 0.0051699
4: -0.0041051, 0.0088404, -0.0034886, 0.0089296, -0.0130347, 0.0123290
5: -0.0012824, 0.0105721, -0.0010690, 0.0106352, -0.0119176, 0.0116412
6: -0.0047069, 0.0030570, -0.0037546, 0.0031455, -0.0078524, 0.0068116
7: -0.0112246, -0.0014692, -0.0112700, -0.0018090, -0.0094156, 0.0098008
8: -0.0113357, 0.0155796, -0.0108608, 0.0157278, -0.0268771, 0.0262504
9: -0.0091672, 0.0063130, -0.0092495, 0.0060345, -0.0152016, 0.0155625

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122341, upper bound: 0.0121383
time: 3.38 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122628, upper bound: 0.0122273
time: 3.06 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0045276, 0.0028078, -0.0041496, 0.0030145, -0.0075421, 0.0069574
1: 0.9893818, 1.0060053, 0.9888607, 1.0052546, -0.0158728, 0.0171446
2: -0.0131884, 0.0035510, -0.0140949, 0.0031595, -0.0156942, 0.0169951
3: 0.0004469, 0.0057479, 0.0005527, 0.0059641, -0.0055172, 0.0051952
4: -0.0041051, 0.0088404, -0.0036075, 0.0095569, -0.0136620, 0.0124479
5: -0.0012824, 0.0105721, -0.0011102, 0.0110787, -0.0123610, 0.0116823
6: -0.0047069, 0.0030570, -0.0040136, 0.0037678, -0.0084747, 0.0070706
7: -0.0112246, -0.0014692, -0.0115893, -0.0017435, -0.0094811, 0.0101201
8: -0.0113357, 0.0155796, -0.0109523, 0.0167707, -0.0279156, 0.0263357
9: -0.0091672, 0.0063130, -0.0098286, 0.0060882, -0.0152554, 0.0161417

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122341, upper bound: 0.0121803
time: 2.93 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0122628, upper bound: 0.0122497
time: 2.81 seconds

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

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0120873, upper bound: 0.0122092
time: 2.80 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0120869, upper bound: 0.0122808
time: 2.77 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0030213, 0.0025732, -0.0042364, 0.0028202, -0.0058415, 0.0068096
1: 0.9899734, 1.0030142, 0.9893506, 1.0054271, -0.0154537, 0.0136636
2: -0.0121592, 0.0019909, -0.0132428, 0.0032494, -0.0147376, 0.0145668
3: 0.0008686, 0.0055025, 0.0005284, 0.0057609, -0.0048923, 0.0049741
4: -0.0021225, 0.0080270, -0.0037219, 0.0088834, -0.0110059, 0.0117488
5: -0.0005963, 0.0099971, -0.0011497, 0.0106026, -0.0111988, 0.0111468
6: -0.0033417, 0.0022500, -0.0041149, 0.0030997, -0.0064413, 0.0063648
7: -0.0108105, -0.0025619, -0.0112465, -0.0016805, -0.0091301, 0.0086846
8: -0.0098084, 0.0142273, -0.0110404, 0.0156510, -0.0252628, 0.0250700
9: -0.0084162, 0.0054171, -0.0092069, 0.0061399, -0.0145561, 0.0146240

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121338, upper bound: 0.0122088
time: 2.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0121338, upper bound: 0.0123050
time: 2.65 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.63 + 596.64 = 601.28 seconds
