## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 12.3086572218


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819)
1: (-6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910)
2: (-7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517)
3: (-8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391)
4: (-8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165)
5: (-7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348)
6: (-6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664)
7: (-7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242)
8: (-10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918)
9: (-6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.66 + 5.05 = 6.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -12.3209782, upper bound: 12.3209782

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209479, upper bound: 12.3209502
time: 4.28 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209460, upper bound: 12.3209460
time: 2.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.18 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.18
Output dim: 8, lower bound: -12.3209479, upper bound: 12.3209502
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.18
Output dim: 8, lower bound: -12.3209460, upper bound: 12.3209460

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -6.8240829, 5.5771694, -7.0235176, 5.7524662, -12.5765457, 12.6006870
1: -6.0102949, 5.0247502, -6.1864815, 5.1664095, -11.1767035, 11.2112312
2: -7.6687365, 5.0140209, -7.9090343, 5.1656170, -12.8343534, 12.9230556
3: -8.2226067, 4.1078248, -8.4604549, 4.2214842, -12.4440908, 12.5682793
4: -8.1181736, 5.5820365, -8.3454752, 5.7502418, -13.8684158, 13.9275112
5: -6.9425273, 5.5226936, -7.1336541, 5.6869812, -12.6295061, 12.6563473
6: -6.2002292, 6.2560167, -6.3902140, 6.4361529, -12.6363821, 12.6462307
7: -6.9225917, 6.7518883, -7.1255641, 6.9407606, -13.8633518, 13.8774519
8: -10.0793161, 4.3257952, -10.3838263, 4.4998660, -14.5791817, 14.7096214
9: -6.0891914, 6.1457334, -6.2737536, 6.3332019, -12.4223938, 12.4194860

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209459, upper bound: 12.3209459
time: 3.30 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209459, upper bound: 12.3209459
time: 4.50 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -8.4467106, 6.8592362, -6.9098301, 5.6536598, -14.1003704, 13.7690659
1: -7.5243611, 6.1799479, -6.0843062, 5.0855155, -12.6098766, 12.2642536
2: -9.5779924, 6.1184621, -7.7728796, 5.0803108, -14.6583033, 13.8913422
3: -10.2551126, 5.0135622, -8.3228855, 4.1566300, -14.4117413, 13.3364477
4: -10.0881882, 6.8794599, -8.2139454, 5.6549449, -15.7431335, 15.0934048
5: -8.5949430, 6.7904534, -7.0235786, 5.5934439, -14.1883841, 13.8140316
6: -7.6619344, 7.6517954, -6.2822556, 6.3332381, -13.9951725, 13.9340515
7: -8.6093159, 8.3841286, -7.0089536, 6.8307657, -15.4400816, 15.3930817
8: -12.4761162, 5.1991315, -10.2132730, 4.4074149, -16.8835297, 15.4124041
9: -7.5354133, 7.6007175, -6.1681223, 6.2268505, -13.7622623, 13.7688398

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208185, upper bound: 12.3208186
time: 6.06 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208108, upper bound: 12.3208167
time: 4.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 12.10 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 12.10
Output dim: 8, lower bound: -12.3209459, upper bound: 12.3209459
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 12.10
Output dim: 8, lower bound: -12.3209459, upper bound: 12.3209459
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 12.10
Output dim: 8, lower bound: -12.3208185, upper bound: 12.3208186
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 12.10
Output dim: 8, lower bound: -12.3208108, upper bound: 12.3208167

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -6.8240829, 5.5771694, -6.8240829, 5.5771694, -12.4012508, 12.4012527
1: -6.0102949, 5.0247502, -6.0102949, 5.0247502, -11.0350447, 11.0350447
2: -7.6687365, 5.0140209, -7.6687365, 5.0140209, -12.6827555, 12.6827555
3: -8.2226067, 4.1078248, -8.2226067, 4.1078248, -12.3304310, 12.3304310
4: -8.1181736, 5.5820365, -8.1181736, 5.5820365, -13.7002087, 13.7002087
5: -6.9425273, 5.5226936, -6.9425273, 5.5226936, -12.4652195, 12.4652195
6: -6.2002292, 6.2560167, -6.2002292, 6.2560167, -12.4562454, 12.4562454
7: -6.9225917, 6.7518883, -6.9225917, 6.7518883, -13.6744785, 13.6744795
8: -10.0793161, 4.3257952, -10.0793161, 4.3257952, -14.4051113, 14.4051113
9: -6.0891914, 6.1457334, -6.0891914, 6.1457334, -12.2349234, 12.2349234

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208203, upper bound: 12.3208243
time: 4.99 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208187, upper bound: 12.3208242
time: 4.17 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -6.8240829, 5.5771694, -8.4467106, 6.8592362, -13.6833181, 14.0238800
1: -6.0102949, 5.0247502, -7.5243611, 6.1799479, -12.1902428, 12.5491114
2: -7.6687365, 5.0140209, -9.5779924, 6.1184621, -13.7871971, 14.5920134
3: -8.2226067, 4.1078248, -10.2551126, 5.0135622, -13.2361689, 14.3629370
4: -8.1181736, 5.5820365, -10.0881882, 6.8794599, -14.9976311, 15.6702251
5: -6.9425273, 5.5226936, -8.5949430, 6.7904534, -13.7329798, 14.1176367
6: -6.2002292, 6.2560167, -7.6619344, 7.6517954, -13.8520241, 13.9179516
7: -6.9225917, 6.7518883, -8.6093159, 8.3841286, -15.3067198, 15.3612041
8: -10.0793161, 4.3257952, -12.4761162, 5.1991315, -15.2784481, 16.8019104
9: -6.0891914, 6.1457334, -7.5354133, 7.6007175, -13.6899090, 13.6811466

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208203, upper bound: 12.3208246
time: 4.84 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208187, upper bound: 12.3208239
time: 3.68 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -8.0714006, 6.5591059, -5.0768452, 4.1851549, -12.2565536, 11.6359510
1: -7.1787357, 5.9146109, -4.3973360, 3.7832766, -10.9620123, 10.3119469
2: -9.1386662, 5.8623695, -5.6196265, 3.8217425, -12.9604092, 11.4819965
3: -9.7906065, 4.8037453, -6.0478468, 3.1262593, -12.9168653, 10.8515911
4: -9.6401291, 6.5797849, -6.0172696, 4.1840906, -13.8242188, 12.5970535
5: -8.2115002, 6.4939218, -5.1371932, 4.2221403, -12.4336405, 11.6311150
6: -7.3189964, 7.3242755, -4.5995159, 4.7149601, -12.0339565, 11.9237919
7: -8.2245560, 8.0108681, -5.1241646, 5.0051970, -13.2297535, 13.1350298
8: -11.9255428, 4.9816918, -7.5083652, 3.3959327, -15.3214760, 12.4900570
9: -7.2017770, 7.2651997, -4.5317354, 4.5812764, -11.7830534, 11.7969332

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208108, upper bound: 12.3208103
time: 4.07 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208108, upper bound: 12.3208150
time: 4.23 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8.1484737, 6.6199622, -6.2110529, 5.0996151, -13.2480879, 12.8310146
1: -7.2503376, 5.9690881, -5.4360266, 4.5905776, -11.8409138, 11.4051142
2: -9.2279568, 5.9143653, -6.9487743, 4.6116214, -13.8395767, 12.8631382
3: -9.8859987, 4.8467669, -7.4514356, 3.7649648, -13.6509638, 12.2982025
4: -9.7332830, 6.6396356, -7.3729525, 5.0925288, -14.8258114, 14.0125885
5: -8.2913876, 6.5536733, -6.3066359, 5.0822473, -13.3736334, 12.8603077
6: -7.3867874, 7.3902183, -5.6444478, 5.7252884, -13.1120758, 13.0346651
7: -8.3038187, 8.0882854, -6.2869229, 6.1310763, -14.4348946, 14.3752079
8: -12.0376520, 5.0223393, -9.1900120, 4.0460644, -16.0837097, 14.2123508
9: -7.2698579, 7.3331351, -5.5479126, 5.6012263, -12.8710842, 12.8810463

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208150, upper bound: 12.3208103
time: 4.77 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208150, upper bound: 12.3208167
time: 4.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 11.27 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 11.27
Output dim: 8, lower bound: -12.3208203, upper bound: 12.3208243
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 11.27
Output dim: 8, lower bound: -12.3208187, upper bound: 12.3208242
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 11.27
Output dim: 8, lower bound: -12.3208203, upper bound: 12.3208246
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 11.27
Output dim: 8, lower bound: -12.3208187, upper bound: 12.3208239
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 11.27
Output dim: 8, lower bound: -12.3208108, upper bound: 12.3208103
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 11.27
Output dim: 8, lower bound: -12.3208108, upper bound: 12.3208150
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 11.27
Output dim: 8, lower bound: -12.3208150, upper bound: 12.3208103
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 11.27
Output dim: 8, lower bound: -12.3208150, upper bound: 12.3208167

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.9937305, 4.1106253, -6.4604583, 5.2852888, -10.2790194, 10.5710831
1: -4.3249941, 3.7235610, -5.6750069, 4.7662492, -9.0912428, 9.3985653
2: -5.5168214, 3.7579212, -7.2411890, 4.7646880, -10.2815094, 10.9991083
3: -5.9486189, 3.0790124, -7.7708092, 3.9033277, -9.8519459, 10.8498201
4: -5.9233742, 4.1134143, -7.6822238, 5.2901173, -11.2134886, 11.7956381
5: -5.0578213, 4.1587343, -6.5685706, 5.2472558, -10.3050766, 10.7273045
6: -4.5201321, 4.6408486, -5.8666525, 5.9362426, -10.4563742, 10.5075016
7: -5.0393720, 4.9274111, -6.5484395, 6.3889561, -11.4283266, 11.4758511
8: -7.3781786, 3.3124800, -9.5426693, 4.1243935, -11.5025711, 12.8551474
9: -4.4551797, 4.5017738, -5.7644114, 5.8193016, -10.2744808, 10.2661839

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208284, upper bound: 12.3208284
time: 3.85 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208284, upper bound: 12.3208288
time: 4.31 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.1172123, 5.0149798, -6.5345097, 5.3434019, -11.4606133, 11.5494900
1: -5.3543868, 4.5219927, -5.7438350, 4.8187828, -10.1731682, 10.2658272
2: -6.8318043, 4.5333023, -7.3266611, 4.8149953, -11.6467991, 11.8599634
3: -7.3414860, 3.7100358, -7.8625102, 3.9447925, -11.2862778, 11.5725460
4: -7.2653461, 5.0101314, -7.7722998, 5.3476195, -12.6129637, 12.7824306
5: -6.2190733, 5.0091400, -6.6460171, 5.3001351, -11.5192070, 11.6551561
6: -5.5515132, 5.6412096, -5.9317164, 5.9994278, -11.5509396, 11.5729256
7: -6.1878991, 6.0418177, -6.6245894, 6.4632545, -12.6511536, 12.6664066
8: -9.0436277, 3.9537547, -9.6512756, 4.1598172, -13.2034435, 13.6050301
9: -5.4574904, 5.5141721, -5.8300071, 5.8847041, -11.3421946, 11.3441782

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208286, upper bound: 12.3208329
time: 4.05 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208286, upper bound: 12.3208368
time: 4.69 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.9937305, 4.1106253, -8.0714006, 6.5591059, -11.5528364, 12.1820250
1: -4.3249941, 3.7235610, -7.1787357, 5.9146109, -10.2396040, 10.9022961
2: -5.5168214, 3.7579212, -9.1386662, 5.8623695, -11.3791904, 12.8965874
3: -5.9486189, 3.0790124, -9.7906065, 4.8037453, -10.7523632, 12.8696184
4: -5.9233742, 4.1134143, -9.6401291, 6.5797849, -12.5031576, 13.7535439
5: -5.0578213, 4.1587343, -8.2115002, 6.4939218, -11.5517426, 12.3702345
6: -4.5201321, 4.6408486, -7.3189964, 7.3242755, -11.8444080, 11.9598427
7: -5.0393720, 4.9274111, -8.2245560, 8.0108681, -13.0502367, 13.1519661
8: -7.3781786, 3.3124800, -11.9255428, 4.9816918, -12.3598700, 15.2380228
9: -4.4551797, 4.5017738, -7.2017770, 7.2651997, -11.7203789, 11.7035503

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208128, upper bound: 12.3208177
time: 4.82 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208128, upper bound: 12.3208177
time: 4.75 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.1172123, 5.0149798, -8.1484737, 6.6199622, -12.7371731, 13.1634541
1: -5.3543868, 4.5219927, -7.2503376, 5.9690881, -11.3234749, 11.7723293
2: -6.8318043, 4.5333023, -9.2279568, 5.9143653, -12.7461691, 13.7612591
3: -7.3414860, 3.7100358, -9.8859987, 4.8467669, -12.1882534, 13.5960331
4: -7.2653461, 5.0101314, -9.7332830, 6.6396356, -13.9049797, 14.7434139
5: -6.2190733, 5.0091400, -8.2913876, 6.5536733, -12.7727461, 13.3005276
6: -5.5515132, 5.6412096, -7.3867874, 7.3902183, -12.9417286, 13.0279970
7: -6.1878991, 6.0418177, -8.3038187, 8.0882854, -14.2761841, 14.3456345
8: -9.0436277, 3.9537547, -12.0376520, 5.0223393, -14.0659676, 15.9914064
9: -5.4574904, 5.5141721, -7.2698579, 7.3331351, -12.7906256, 12.7840271

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208128, upper bound: 12.3208218
time: 5.45 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208128, upper bound: 12.3208242
time: 4.51 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.5307431, 5.3273740, -5.0768452, 4.1851549, -10.7158966, 10.4042187
1: -5.7598782, 4.8238630, -4.3973360, 3.7832766, -9.5431547, 9.2211990
2: -7.3339195, 4.8094854, -5.6196265, 3.8217425, -11.1556616, 10.4291105
3: -7.8829160, 3.9413977, -6.0478468, 3.1262593, -11.0091753, 9.9892445
4: -7.7990632, 5.3482261, -6.0172696, 4.1840906, -11.9831543, 11.3654957
5: -6.6346936, 5.2755542, -5.1371932, 4.2221403, -10.8568344, 10.4127474
6: -5.9102049, 5.9731164, -4.5995159, 4.7149601, -10.6251650, 10.5726318
7: -6.6444540, 6.4782176, -5.1241646, 5.0051970, -11.6496496, 11.6023827
8: -9.6615982, 4.1045771, -7.5083652, 3.3959327, -13.0575304, 11.6129417
9: -5.8306189, 5.8863440, -4.5317354, 4.5812764, -10.4118938, 10.4180775

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208185, upper bound: 12.3208179
time: 3.94 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208180, upper bound: 12.3208179
time: 3.82 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.7105088, 6.2758937, -5.0768452, 4.1851549, -11.8956642, 11.3527384
1: -6.8419647, 5.6589756, -4.3973360, 3.7832766, -10.6252413, 10.0563116
2: -8.7103539, 5.6195402, -5.6196265, 3.8217425, -12.5320959, 11.2391663
3: -9.3404226, 4.6019154, -6.0478468, 3.1262593, -12.4666824, 10.6497612
4: -9.2030525, 6.2869210, -6.0172696, 4.1840906, -13.3871422, 12.3041897
5: -7.8467627, 6.2111130, -5.1371932, 4.2221403, -12.0689030, 11.3483067
6: -6.9894147, 7.0163298, -4.5995159, 4.7149601, -11.7043743, 11.6158438
7: -7.8471293, 7.6470318, -5.1241646, 5.0051970, -12.8523264, 12.7711964
8: -11.4015112, 4.7855611, -7.5083652, 3.3959327, -14.7974424, 12.2939243
9: -6.8802962, 6.9447989, -4.5317354, 4.5812764, -11.4615717, 11.4765310

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208185, upper bound: 12.3208182
time: 4.54 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208180, upper bound: 12.3208182
time: 3.10 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.5307431, 5.3273740, -6.2110529, 5.0996151, -11.6303568, 11.5384254
1: -5.7598782, 4.8238630, -5.4360266, 4.5905776, -10.3504562, 10.2598877
2: -7.3339195, 4.8094854, -6.9487743, 4.6116214, -11.9455395, 11.7582588
3: -7.8829160, 3.9413977, -7.4514356, 3.7649648, -11.6478806, 11.3928337
4: -7.7990632, 5.3482261, -7.3729525, 5.0925288, -12.8915920, 12.7211771
5: -6.6346936, 5.2755542, -6.3066359, 5.0822473, -11.7169409, 11.5821886
6: -5.9102049, 5.9731164, -5.6444478, 5.7252884, -11.6354933, 11.6175642
7: -6.6444540, 6.4782176, -6.2869229, 6.1310763, -12.7755289, 12.7651405
8: -9.6615982, 4.1045771, -9.1900120, 4.0460644, -13.7076626, 13.2945890
9: -5.8306189, 5.8863440, -5.5479126, 5.6012263, -11.4318447, 11.4342537

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208103, upper bound: 12.3208103
time: 3.66 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208108, upper bound: 12.3208108
time: 4.86 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.7105088, 6.2758937, -6.2110529, 5.0996151, -12.8101234, 12.4869442
1: -6.8419647, 5.6589756, -5.4360266, 4.5905776, -11.4325428, 11.0950012
2: -8.7103539, 5.6195402, -6.9487743, 4.6116214, -13.3219757, 12.5683136
3: -9.3404226, 4.6019154, -7.4514356, 3.7649648, -13.1053867, 12.0533504
4: -9.2030525, 6.2869210, -7.3729525, 5.0925288, -14.2955818, 13.6598740
5: -7.8467627, 6.2111130, -6.3066359, 5.0822473, -12.9290094, 12.5177488
6: -6.9894147, 7.0163298, -5.6444478, 5.7252884, -12.7147017, 12.6607761
7: -7.8471293, 7.6470318, -6.2869229, 6.1310763, -13.9782038, 13.9339542
8: -11.4015112, 4.7855611, -9.1900120, 4.0460644, -15.4475756, 13.9755726
9: -6.8802962, 6.9447989, -5.5479126, 5.6012263, -12.4815216, 12.4927082

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 250

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208103, upper bound: 12.3208167
time: 4.10 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208108, upper bound: 12.3208163
time: 4.22 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 10.56 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 10.56
Output dim: 8, lower bound: -12.3208284, upper bound: 12.3208284
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 10.56
Output dim: 8, lower bound: -12.3208284, upper bound: 12.3208288
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.56
Output dim: 8, lower bound: -12.3208286, upper bound: 12.3208329
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.56
Output dim: 8, lower bound: -12.3208286, upper bound: 12.3208368
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 10.56
Output dim: 8, lower bound: -12.3208128, upper bound: 12.3208177
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 10.56
Output dim: 8, lower bound: -12.3208128, upper bound: 12.3208177
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.56
Output dim: 8, lower bound: -12.3208128, upper bound: 12.3208218
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.56
Output dim: 8, lower bound: -12.3208128, upper bound: 12.3208242
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 10.56
Output dim: 8, lower bound: -12.3208185, upper bound: 12.3208179
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 10.56
Output dim: 8, lower bound: -12.3208180, upper bound: 12.3208179
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.56
Output dim: 8, lower bound: -12.3208185, upper bound: 12.3208182
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.56
Output dim: 8, lower bound: -12.3208180, upper bound: 12.3208182
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 10.56
Output dim: 8, lower bound: -12.3208103, upper bound: 12.3208103
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 10.56
Output dim: 8, lower bound: -12.3208108, upper bound: 12.3208108
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.56
Output dim: 8, lower bound: -12.3208103, upper bound: 12.3208167
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.56
Output dim: 8, lower bound: -12.3208108, upper bound: 12.3208163

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.9937305, 4.1106253, -4.9937305, 4.1106253, -9.1043549, 9.1043549
1: -4.3249941, 3.7235610, -4.3249941, 3.7235610, -8.0485544, 8.0485535
2: -5.5168214, 3.7579212, -5.5168214, 3.7579212, -9.2747421, 9.2747421
3: -5.9486189, 3.0790124, -5.9486189, 3.0790124, -9.0276318, 9.0276308
4: -5.9233742, 4.1134143, -5.9233742, 4.1134143, -10.0367889, 10.0367889
5: -5.0578213, 4.1587343, -5.0578213, 4.1587343, -9.2165556, 9.2165546
6: -4.5201321, 4.6408486, -4.5201321, 4.6408486, -9.1609783, 9.1609793
7: -5.0393720, 4.9274111, -5.0393720, 4.9274111, -9.9667816, 9.9667835
8: -7.3781786, 3.3124800, -7.3781786, 3.3124800, -10.6906586, 10.6906586
9: -4.4551797, 4.5017738, -4.4551797, 4.5017738, -8.9569530, 8.9569530

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206399, upper bound: 12.3205677
time: 3.05 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208020, upper bound: 12.3208007
time: 4.24 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.9937305, 4.1106253, -6.1172123, 5.0149798, -10.0087109, 10.2278366
1: -4.3249941, 3.7235610, -5.3543868, 4.5219927, -8.8469868, 9.0779476
2: -5.5168214, 3.7579212, -6.8318043, 4.5333023, -10.0501232, 10.5897236
3: -5.9486189, 3.0790124, -7.3414860, 3.7100358, -9.6586533, 10.4204979
4: -5.9233742, 4.1134143, -7.2653461, 5.0101314, -10.9335041, 11.3787603
5: -5.0578213, 4.1587343, -6.2190733, 5.0091400, -10.0669613, 10.3778076
6: -4.5201321, 4.6408486, -5.5515132, 5.6412096, -10.1613417, 10.1923599
7: -5.0393720, 4.9274111, -6.1878991, 6.0418177, -11.0811901, 11.1153088
8: -7.3781786, 3.3124800, -9.0436277, 3.9537547, -11.3319330, 12.3561058
9: -4.4551797, 4.5017738, -5.4574904, 5.5141721, -9.9693518, 9.9592648

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206399, upper bound: 12.3205677
time: 3.47 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208020, upper bound: 12.3208007
time: 4.09 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6.1172123, 5.0149798, -4.9937305, 4.1106253, -10.2278376, 10.0087109
1: -5.3543868, 4.5219927, -4.3249941, 3.7235610, -9.0779467, 8.8469868
2: -6.8318043, 4.5333023, -5.5168214, 3.7579212, -10.5897236, 10.0501232
3: -7.3414860, 3.7100358, -5.9486189, 3.0790124, -10.4204979, 9.6586542
4: -7.2653461, 5.0101314, -5.9233742, 4.1134143, -11.3787594, 10.9335051
5: -6.2190733, 5.0091400, -5.0578213, 4.1587343, -10.3778076, 10.0669603
6: -5.5515132, 5.6412096, -4.5201321, 4.6408486, -10.1923599, 10.1613407
7: -6.1878991, 6.0418177, -5.0393720, 4.9274111, -11.1153088, 11.0811901
8: -9.0436277, 3.9537547, -7.3781786, 3.3124800, -12.3561058, 11.3319330
9: -5.4574904, 5.5141721, -4.4551797, 4.5017738, -9.9592648, 9.9693518

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206294, upper bound: 12.3205625
time: 3.31 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207963, upper bound: 12.3208030
time: 3.38 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6.1172123, 5.0149798, -6.1172123, 5.0149798, -11.1321907, 11.1321917
1: -5.3543868, 4.5219927, -5.3543868, 4.5219927, -9.8763790, 9.8763790
2: -6.8318043, 4.5333023, -6.8318043, 4.5333023, -11.3651056, 11.3651056
3: -7.3414860, 3.7100358, -7.3414860, 3.7100358, -11.0515213, 11.0515213
4: -7.2653461, 5.0101314, -7.2653461, 5.0101314, -12.2754745, 12.2754765
5: -6.2190733, 5.0091400, -6.2190733, 5.0091400, -11.2282133, 11.2282133
6: -5.5515132, 5.6412096, -5.5515132, 5.6412096, -11.1927223, 11.1927214
7: -6.1878991, 6.0418177, -6.1878991, 6.0418177, -12.2297153, 12.2297153
8: -9.0436277, 3.9537547, -9.0436277, 3.9537547, -12.9973822, 12.9973822
9: -5.4574904, 5.5141721, -5.4574904, 5.5141721, -10.9716625, 10.9716625

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206294, upper bound: 12.3205734
time: 4.95 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207957, upper bound: 12.3208096
time: 3.84 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.9937305, 4.1106253, -6.5307431, 5.3273740, -10.3211040, 10.6413679
1: -4.3249941, 3.7235610, -5.7598782, 4.8238630, -9.1488571, 9.4834394
2: -5.5168214, 3.7579212, -7.3339195, 4.8094854, -10.3263063, 11.0918398
3: -5.9486189, 3.0790124, -7.8829160, 3.9413977, -9.8900166, 10.9619274
4: -5.9233742, 4.1134143, -7.7990632, 5.3482261, -11.2715979, 11.9124775
5: -5.0578213, 4.1587343, -6.6346936, 5.2755542, -10.3333740, 10.7934284
6: -4.5201321, 4.6408486, -5.9102049, 5.9731164, -10.4932480, 10.5510530
7: -5.0393720, 4.9274111, -6.6444540, 6.4782176, -11.5175896, 11.5718651
8: -7.3781786, 3.3124800, -9.6615982, 4.1045771, -11.4827557, 12.9740763
9: -4.4551797, 4.5017738, -5.8306189, 5.8863440, -10.3415232, 10.3323917

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206096, upper bound: 12.3205431
time: 4.93 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207833, upper bound: 12.3207894
time: 3.36 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.9937305, 4.1106253, -7.7105088, 6.2758937, -11.2696238, 11.8211346
1: -4.3249941, 3.7235610, -6.8419647, 5.6589756, -9.9839678, 10.5655251
2: -5.5168214, 3.7579212, -8.7103539, 5.6195402, -11.1363611, 12.4682751
3: -5.9486189, 3.0790124, -9.3404226, 4.6019154, -10.5505333, 12.4194345
4: -5.9233742, 4.1134143, -9.2030525, 6.2869210, -12.2102928, 13.3164673
5: -5.0578213, 4.1587343, -7.8467627, 6.2111130, -11.2689342, 12.0054970
6: -4.5201321, 4.6408486, -6.9894147, 7.0163298, -11.5364599, 11.6302624
7: -5.0393720, 4.9274111, -7.8471293, 7.6470318, -12.6864014, 12.7745399
8: -7.3781786, 3.3124800, -11.4015112, 4.7855611, -12.1637402, 14.7139902
9: -4.4551797, 4.5017738, -6.8802962, 6.9447989, -11.3999767, 11.3820705

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206096, upper bound: 12.3205431
time: 4.86 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207833, upper bound: 12.3207894
time: 3.37 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6.1172123, 5.0149798, -6.5307431, 5.3273740, -11.4445858, 11.5457230
1: -5.3543868, 4.5219927, -5.7598782, 4.8238630, -10.1782494, 10.2818708
2: -6.8318043, 4.5333023, -7.3339195, 4.8094854, -11.6412878, 11.8672218
3: -7.3414860, 3.7100358, -7.8829160, 3.9413977, -11.2828836, 11.5929508
4: -7.2653461, 5.0101314, -7.7990632, 5.3482261, -12.6135702, 12.8091946
5: -6.2190733, 5.0091400, -6.6346936, 5.2755542, -11.4946270, 11.6438332
6: -5.5515132, 5.6412096, -5.9102049, 5.9731164, -11.5246296, 11.5514145
7: -6.1878991, 6.0418177, -6.6444540, 6.4782176, -12.6661167, 12.6862717
8: -9.0436277, 3.9537547, -9.6615982, 4.1045771, -13.1482048, 13.6153526
9: -5.4574904, 5.5141721, -5.8306189, 5.8863440, -11.3438339, 11.3447876

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205981, upper bound: 12.3205362
time: 4.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207773, upper bound: 12.3207915
time: 3.02 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.1172123, 5.0149798, -7.7105088, 6.2758937, -12.3931036, 12.7254887
1: -5.3543868, 4.5219927, -6.8419647, 5.6589756, -11.0133619, 11.3639574
2: -6.8318043, 4.5333023, -8.7103539, 5.6195402, -12.4513426, 13.2436562
3: -7.3414860, 3.7100358, -9.3404226, 4.6019154, -11.9434013, 13.0504580
4: -7.2653461, 5.0101314, -9.2030525, 6.2869210, -13.5522633, 14.2131834
5: -6.2190733, 5.0091400, -7.8467627, 6.2111130, -12.4301863, 12.8559027
6: -5.5515132, 5.6412096, -6.9894147, 7.0163298, -12.5678415, 12.6306229
7: -6.1878991, 6.0418177, -7.8471293, 7.6470318, -13.8349295, 13.8889465
8: -9.0436277, 3.9537547, -11.4015112, 4.7855611, -13.8291893, 15.3552656
9: -5.4574904, 5.5141721, -6.8802962, 6.9447989, -12.4022884, 12.3944664

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205981, upper bound: 12.3205463
time: 4.54 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207773, upper bound: 12.3207958
time: 3.97 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.5307431, 5.3273740, -4.9937305, 4.1106253, -10.6413660, 10.3211040
1: -5.7598782, 4.8238630, -4.3249941, 3.7235610, -9.4834394, 9.1488571
2: -7.3339195, 4.8094854, -5.5168214, 3.7579212, -11.0918388, 10.3263054
3: -7.8829160, 3.9413977, -5.9486189, 3.0790124, -10.9619265, 9.8900166
4: -7.7990632, 5.3482261, -5.9233742, 4.1134143, -11.9124775, 11.2715979
5: -6.6346936, 5.2755542, -5.0578213, 4.1587343, -10.7934284, 10.3333750
6: -5.9102049, 5.9731164, -4.5201321, 4.6408486, -10.5510521, 10.4932470
7: -6.6444540, 6.4782176, -5.0393720, 4.9274111, -11.5718651, 11.5175896
8: -9.6615982, 4.1045771, -7.3781786, 3.3124800, -12.9740772, 11.4827557
9: -5.8306189, 5.8863440, -4.4551797, 4.5017738, -10.3323917, 10.3415232

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206325, upper bound: 12.3205540
time: 3.75 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207977, upper bound: 12.3207971
time: 3.79 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.5307431, 5.3273740, -6.4879503, 5.2926636, -11.8234062, 11.8153238
1: -5.7598782, 4.8238630, -5.7203236, 4.7929153, -10.5527925, 10.5441856
2: -7.3339195, 4.8094854, -7.2848334, 4.7814498, -12.1153679, 12.0943184
3: -7.8829160, 3.9413977, -7.8311834, 3.9167318, -11.7996483, 11.7725811
4: -7.7990632, 5.3482261, -7.7490311, 5.3102360, -13.1092987, 13.0972538
5: -6.6346936, 5.2755542, -6.5929055, 5.2412081, -11.8759022, 11.8684578
6: -5.9102049, 5.9731164, -5.8677812, 5.9376740, -11.8478775, 11.8408976
7: -6.6444540, 6.4782176, -6.5998178, 6.4354320, -13.0798855, 13.0780354
8: -9.6615982, 4.1045771, -9.5999622, 4.0830383, -13.7446365, 13.7045393
9: -5.8306189, 5.8863440, -5.7914481, 5.8493953, -11.6800137, 11.6777916

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206325, upper bound: 12.3205540
time: 10.03 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207977, upper bound: 12.3207971
time: 4.80 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.7105088, 6.2758937, -4.9937305, 4.1106253, -11.8211336, 11.2696238
1: -6.8419647, 5.6589756, -4.3249941, 3.7235610, -10.5655260, 9.9839687
2: -8.7103539, 5.6195402, -5.5168214, 3.7579212, -12.4682751, 11.1363611
3: -9.3404226, 4.6019154, -5.9486189, 3.0790124, -12.4194336, 10.5505333
4: -9.2030525, 6.2869210, -5.9233742, 4.1134143, -13.3164673, 12.2102928
5: -7.8467627, 6.2111130, -5.0578213, 4.1587343, -12.0054970, 11.2689342
6: -6.9894147, 7.0163298, -4.5201321, 4.6408486, -11.6302624, 11.5364599
7: -7.8471293, 7.6470318, -5.0393720, 4.9274111, -12.7745399, 12.6864014
8: -11.4015112, 4.7855611, -7.3781786, 3.3124800, -14.7139893, 12.1637383
9: -6.8802962, 6.9447989, -4.4551797, 4.5017738, -11.3820705, 11.3999777

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205904, upper bound: 12.3205121
time: 4.30 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205904, upper bound: 12.3207829
time: 5.82 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.7105088, 6.2758937, -6.4879503, 5.2926636, -13.0031719, 12.7638416
1: -6.8419647, 5.6589756, -5.7203236, 4.7929153, -11.6348791, 11.3792992
2: -8.7103539, 5.6195402, -7.2848334, 4.7814498, -13.4918041, 12.9043732
3: -9.3404226, 4.6019154, -7.8311834, 3.9167318, -13.2571545, 12.4330969
4: -9.2030525, 6.2869210, -7.7490311, 5.3102360, -14.5132875, 14.0359507
5: -7.8467627, 6.2111130, -6.5929055, 5.2412081, -13.0879707, 12.8040180
6: -6.9894147, 7.0163298, -5.8677812, 5.9376740, -12.9270859, 12.8841095
7: -7.8471293, 7.6470318, -6.5998178, 6.4354320, -14.2825603, 14.2468491
8: -11.4015112, 4.7855611, -9.5999622, 4.0830383, -15.4845486, 14.3855228
9: -6.8802962, 6.9447989, -5.7914481, 5.8493953, -12.7296915, 12.7362461

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205904, upper bound: 12.3205121
time: 7.98 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207817, upper bound: 12.3207834
time: 4.01 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.5307431, 5.3273740, -6.1295633, 5.0242362, -11.5549774, 11.4569378
1: -5.7598782, 4.8238630, -5.3641973, 4.5315752, -10.2914534, 10.1880608
2: -7.3339195, 4.8094854, -6.8468313, 4.5483141, -11.8822327, 11.6563168
3: -7.8829160, 3.9413977, -7.3542814, 3.7168274, -11.5997429, 11.2956791
4: -7.7990632, 5.3482261, -7.2809348, 5.0212455, -12.8203087, 12.6291599
5: -6.6346936, 5.2755542, -6.2283916, 5.0179029, -11.6525965, 11.5039444
6: -5.9102049, 5.9731164, -5.5656548, 5.6500783, -11.5602837, 11.5387707
7: -6.6444540, 6.4782176, -6.2038956, 6.0534964, -12.6979504, 12.6821136
8: -9.6615982, 4.1045771, -9.0624199, 3.9638088, -13.6254063, 13.1669970
9: -5.8306189, 5.8863440, -5.4706750, 5.5227056, -11.3533249, 11.3570194

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206049, upper bound: 12.3205298
time: 5.92 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207834, upper bound: 12.3207817
time: 3.60 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.5307431, 5.3273740, -7.6510921, 6.2332101, -12.7639523, 12.9784660
1: -5.7598782, 4.8238630, -6.7866654, 5.6211877, -11.3810635, 11.6105270
2: -7.3339195, 4.8094854, -8.6465693, 5.5867414, -12.9206600, 13.4560537
3: -7.8829160, 3.9413977, -9.2694921, 4.5718889, -12.4548054, 13.2108898
4: -7.7990632, 5.3482261, -9.1361771, 6.2448330, -14.0438948, 14.4844017
5: -6.6346936, 5.2755542, -7.7900481, 6.1685662, -12.8032598, 13.0656013
6: -5.9102049, 5.9731164, -6.9416251, 6.9691062, -12.8793087, 12.9147396
7: -6.6444540, 6.4782176, -7.7914972, 7.5913353, -14.2357893, 14.2697144
8: -9.6615982, 4.1045771, -11.3226461, 4.7572546, -14.4188519, 15.4272232
9: -5.8306189, 5.8863440, -6.8329139, 6.8935776, -12.7241936, 12.7192574

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206049, upper bound: 12.3205298
time: 4.57 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207834, upper bound: 12.3207817
time: 3.72 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.7105088, 6.2758937, -6.1295633, 5.0242362, -12.7347450, 12.4054565
1: -6.8419647, 5.6589756, -5.3641973, 4.5315752, -11.3735390, 11.0231724
2: -8.7103539, 5.6195402, -6.8468313, 4.5483141, -13.2586679, 12.4663715
3: -9.3404226, 4.6019154, -7.3542814, 3.7168274, -13.0572500, 11.9561968
4: -9.2030525, 6.2869210, -7.2809348, 5.0212455, -14.2242966, 13.5678539
5: -7.8467627, 6.2111130, -6.2283916, 5.0179029, -12.8646631, 12.4395046
6: -6.9894147, 7.0163298, -5.5656548, 5.6500783, -12.6394920, 12.5819836
7: -7.8471293, 7.6470318, -6.2038956, 6.0534964, -13.9006252, 13.8509245
8: -11.4015112, 4.7855611, -9.0624199, 3.9638088, -15.3653164, 13.8479805
9: -6.8802962, 6.9447989, -5.4706750, 5.5227056, -12.4030008, 12.4154739

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205996, upper bound: 12.3205199
time: 3.72 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207844, upper bound: 12.3207855
time: 5.50 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.7105088, 6.2758937, -7.6510921, 6.2332101, -13.9437189, 13.9269848
1: -6.8419647, 5.6589756, -6.7866654, 5.6211877, -12.4631519, 12.4456406
2: -8.7103539, 5.6195402, -8.6465693, 5.5867414, -14.2970953, 14.2661095
3: -9.3404226, 4.6019154, -9.2694921, 4.5718889, -13.9123116, 13.8714075
4: -9.2030525, 6.2869210, -9.1361771, 6.2448330, -15.4478855, 15.4230976
5: -7.8467627, 6.2111130, -7.7900481, 6.1685662, -14.0153284, 14.0011616
6: -6.9894147, 7.0163298, -6.9416251, 6.9691062, -13.9585190, 13.9579525
7: -7.8471293, 7.6470318, -7.7914972, 7.5913353, -15.4384613, 15.4385281
8: -11.4015112, 4.7855611, -11.3226461, 4.7572546, -16.1587639, 16.1082058
9: -6.8802962, 6.9447989, -6.8329139, 6.8935776, -13.7738733, 13.7777119

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205884, upper bound: 12.3205199
time: 6.35 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205996, upper bound: 12.3207855
time: 5.78 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 13.88 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3206399, upper bound: 12.3205677
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3208020, upper bound: 12.3208007
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3206399, upper bound: 12.3205677
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3208020, upper bound: 12.3208007
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3206294, upper bound: 12.3205625
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3207963, upper bound: 12.3208030
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3206294, upper bound: 12.3205734
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3207957, upper bound: 12.3208096
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3206096, upper bound: 12.3205431
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3207833, upper bound: 12.3207894
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3206096, upper bound: 12.3205431
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3207833, upper bound: 12.3207894
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3205981, upper bound: 12.3205362
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3207773, upper bound: 12.3207915
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3205981, upper bound: 12.3205463
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3207773, upper bound: 12.3207958
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3206325, upper bound: 12.3205540
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3207977, upper bound: 12.3207971
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3206325, upper bound: 12.3205540
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3207977, upper bound: 12.3207971
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3205904, upper bound: 12.3205121
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3205904, upper bound: 12.3207829
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3205904, upper bound: 12.3205121
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3207817, upper bound: 12.3207834
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3206049, upper bound: 12.3205298
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3207834, upper bound: 12.3207817
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3206049, upper bound: 12.3205298
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3207834, upper bound: 12.3207817
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3205996, upper bound: 12.3205199
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3207844, upper bound: 12.3207855
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3205884, upper bound: 12.3205199
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -12.3205996, upper bound: 12.3207855

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2.8311906, 2.3220236, -4.2766476, 3.5239797, -6.3551702, 6.5986710
1: -2.2613404, 2.1600585, -3.6465240, 3.2078359, -5.4691763, 5.8065815
2: -2.9074759, 2.2451537, -4.6572561, 3.2555492, -6.1630249, 6.9024091
3: -3.1814883, 1.8378270, -5.0397677, 2.6677334, -5.8492217, 6.8775949
4: -3.2304306, 2.3435440, -5.0384455, 3.5317862, -6.7622166, 7.3819895
5: -2.7489042, 2.5289159, -4.3007135, 3.6246099, -6.3735142, 6.8296294
6: -2.5099192, 2.7699656, -3.8607235, 4.0103245, -6.5202436, 6.6306891
7: -2.7814639, 2.7111759, -4.2816057, 4.2018089, -6.9832726, 6.9927816
8: -4.0767484, 2.2300501, -6.2959847, 2.9197886, -6.9965358, 8.5260353
9: -2.4844346, 2.5108836, -3.8042140, 3.8403230, -6.3247576, 6.3150973

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205839, upper bound: 12.3204993
time: 5.72 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206549, upper bound: 12.3205718
time: 6.13 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.5567398, 3.7537115, -4.9937305, 4.1106253, -8.6673641, 8.7474422
1: -3.9131050, 3.4102268, -4.3249941, 3.7235610, -7.6366663, 7.7352204
2: -4.9935207, 3.4527514, -5.5168214, 3.7579212, -8.7514400, 8.9695730
3: -5.3967738, 2.8288891, -5.9486189, 3.0790124, -8.4757862, 8.7775078
4: -5.3863025, 3.7594538, -5.9233742, 4.1134143, -9.4997168, 9.6828270
5: -4.5990105, 3.8344724, -5.0578213, 4.1587343, -8.7577429, 8.8922920
6: -4.1194143, 4.2569590, -4.5201321, 4.6408486, -8.7602634, 8.7770901
7: -4.5788836, 4.4864898, -5.0393720, 4.9274111, -9.5062923, 9.5258617
8: -6.7199435, 3.0706205, -7.3781786, 3.3124800, -10.0324211, 10.4487991
9: -4.0594816, 4.0996399, -4.4551797, 4.5017738, -8.5612545, 8.5548191

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205866, upper bound: 12.3206585
time: 3.72 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205866, upper bound: 12.3208169
time: 5.23 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2.8311906, 2.3220236, -5.3648233, 4.4151630, -7.2463536, 7.6868458
1: -2.2613404, 2.1600585, -4.6608467, 3.9886897, -6.2500300, 6.8209038
2: -2.9074759, 2.2451537, -5.9478879, 4.0200586, -6.9275346, 8.1930418
3: -3.1814883, 1.8378270, -6.4093800, 3.2885075, -6.4699955, 8.2472067
4: -3.2304306, 2.3435440, -6.3613377, 4.4105120, -7.6409426, 8.7048817
5: -2.7489042, 2.5289159, -5.4451065, 4.4575963, -7.2065005, 7.9740219
6: -2.5099192, 2.7699656, -4.8746610, 4.9919209, -7.5018401, 7.6446266
7: -2.7814639, 2.7111759, -5.4127965, 5.2937651, -8.0752287, 8.1239719
8: -4.0767484, 2.2300501, -7.9360256, 3.5417500, -7.6184983, 10.1660757
9: -2.4844346, 2.5108836, -4.7882199, 4.8384500, -7.3228846, 7.2991037

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205707, upper bound: 12.3204772
time: 8.29 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206428, upper bound: 12.3205540
time: 5.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.5567398, 3.7537115, -6.1172123, 5.0149798, -9.5717201, 9.8709240
1: -3.9131050, 3.4102268, -5.3543868, 4.5219927, -8.4350977, 8.7646141
2: -4.9935207, 3.4527514, -6.8318043, 4.5333023, -9.5268221, 10.2845535
3: -5.3967738, 2.8288891, -7.3414860, 3.7100358, -9.1068077, 10.1703749
4: -5.3863025, 3.7594538, -7.2653461, 5.0101314, -10.3964329, 11.0247984
5: -4.5990105, 3.8344724, -6.2190733, 5.0091400, -9.6081486, 10.0535450
6: -4.1194143, 4.2569590, -5.5515132, 5.6412096, -9.7606239, 9.8084717
7: -4.5788836, 4.4864898, -6.1878991, 6.0418177, -10.6206989, 10.6743889
8: -6.7199435, 3.0706205, -9.0436277, 3.9537547, -10.6736975, 12.1142483
9: -4.0594816, 4.0996399, -5.4574904, 5.5141721, -9.5736523, 9.5571308

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205625, upper bound: 12.3206300
time: 3.01 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205625, upper bound: 12.3208001
time: 4.53 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.8669133, 3.1989429, -4.2766476, 3.5239797, -7.3908930, 7.4755893
1: -3.2499230, 2.9173281, -3.6465240, 3.2078359, -6.4577575, 6.5638509
2: -4.1628628, 2.9817739, -4.6572561, 3.2555492, -7.4184117, 7.6390290
3: -4.5234685, 2.4348857, -5.0397677, 2.6677334, -7.1912022, 7.4746532
4: -4.5215826, 3.2037556, -5.0384455, 3.5317862, -8.0533686, 8.2421999
5: -3.8708894, 3.3401735, -4.3007135, 3.6246099, -7.4954987, 7.6408873
6: -3.5120084, 3.7009213, -3.8607235, 4.0103245, -7.5223322, 7.5616446
7: -3.8463855, 3.7847154, -4.2816057, 4.2018089, -8.0481949, 8.0663204
8: -5.6910677, 2.7353666, -6.2959847, 2.9197886, -8.6108541, 9.0313511
9: -3.4377427, 3.4709694, -3.8042140, 3.8403230, -7.2780657, 7.2751832

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205553, upper bound: 12.3204870
time: 5.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206271, upper bound: 12.3205484
time: 6.66 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.6790013, 4.6648865, -4.9937305, 4.1106253, -9.7896271, 9.6586170
1: -4.9497828, 4.2107515, -4.3249941, 3.7235610, -8.6733427, 8.5357456
2: -6.3152680, 4.2339687, -5.5168214, 3.7579212, -10.0731878, 9.7507896
3: -6.7977176, 3.4640009, -5.9486189, 3.0790124, -9.8767300, 9.4126196
4: -6.7379527, 4.6596866, -5.9233742, 4.1134143, -10.8513670, 10.5830593
5: -5.7692528, 4.6893191, -5.0578213, 4.1587343, -9.9279871, 9.7471399
6: -5.1567168, 5.2623086, -4.5201321, 4.6408486, -9.7975655, 9.7824402
7: -5.7352281, 5.6052494, -5.0393720, 4.9274111, -10.6626377, 10.6446209
8: -8.3961763, 3.7126670, -7.3781786, 3.3124800, -11.7086544, 11.0908442
9: -5.0668244, 5.1196637, -4.4551797, 4.5017738, -9.5685978, 9.5748434

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205677, upper bound: 12.3206439
time: 3.79 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205677, upper bound: 12.3208041
time: 4.63 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.8669133, 3.1989429, -5.3648233, 4.4151630, -8.2820759, 8.5637665
1: -3.2499230, 2.9173281, -4.6608467, 3.9886897, -7.2386127, 7.5781736
2: -4.1628628, 2.9817739, -5.9478879, 4.0200586, -8.1829205, 8.9296618
3: -4.5234685, 2.4348857, -6.4093800, 3.2885075, -7.8119760, 8.8442659
4: -4.5215826, 3.2037556, -6.3613377, 4.4105120, -8.9320946, 9.5650921
5: -3.8708894, 3.3401735, -5.4451065, 4.4575963, -8.3284855, 8.7852793
6: -3.5120084, 3.7009213, -4.8746610, 4.9919209, -8.5039291, 8.5755825
7: -3.8463855, 3.7847154, -5.4127965, 5.2937651, -9.1401501, 9.1975117
8: -5.6910677, 2.7353666, -7.9360256, 3.5417500, -9.2328167, 10.6713915
9: -3.4377427, 3.4709694, -4.7882199, 4.8384500, -8.2761927, 8.2591896

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205791, upper bound: 12.3205081
time: 4.98 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206471, upper bound: 12.3205609
time: 4.19 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.6790013, 4.6648865, -6.1172123, 5.0149798, -10.6939812, 10.7820978
1: -4.9497828, 4.2107515, -5.3543868, 4.5219927, -9.4717751, 9.5651379
2: -6.3152680, 4.2339687, -6.8318043, 4.5333023, -10.8485680, 11.0657711
3: -6.7977176, 3.4640009, -7.3414860, 3.7100358, -10.5077534, 10.8054867
4: -6.7379527, 4.6596866, -7.2653461, 5.0101314, -11.7480831, 11.9250317
5: -5.7692528, 4.6893191, -6.2190733, 5.0091400, -10.7783928, 10.9083920
6: -5.1567168, 5.2623086, -5.5515132, 5.6412096, -10.7979259, 10.8138218
7: -5.7352281, 5.6052494, -6.1878991, 6.0418177, -11.7770443, 11.7931461
8: -8.3961763, 3.7126670, -9.0436277, 3.9537547, -12.3499308, 12.7562943
9: -5.0668244, 5.1196637, -5.4574904, 5.5141721, -10.5809956, 10.5771542

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205735, upper bound: 12.3206508
time: 4.31 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205736, upper bound: 12.3208096
time: 4.67 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2.8311906, 2.3220236, -5.7768011, 4.7242775, -7.5554681, 8.0988235
1: -2.2613404, 2.1600585, -5.0637140, 4.2883568, -6.5496969, 7.2237720
2: -2.9074759, 2.2451537, -6.4475064, 4.2951202, -7.2025962, 8.6926584
3: -3.1814883, 1.8378270, -6.9483709, 3.5184388, -6.6999264, 8.7861977
4: -3.2304306, 2.3435440, -6.8928194, 4.7463875, -7.9768181, 9.2363634
5: -2.7489042, 2.5289159, -5.8592215, 4.7174950, -7.4663992, 8.3881378
6: -2.5099192, 2.7699656, -5.2313027, 5.3176832, -7.8276024, 8.0012684
7: -2.7814639, 2.7111759, -5.8675890, 5.7272558, -8.5087194, 8.5787630
8: -4.0767484, 2.2300501, -8.5508451, 3.6901088, -7.7668571, 10.7808952
9: -2.4844346, 2.5108836, -5.1589427, 5.2090325, -7.6934671, 7.6698256

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205520, upper bound: 12.3204571
time: 3.88 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206288, upper bound: 12.3205517
time: 3.59 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.5567398, 3.7537115, -6.5307431, 5.3273740, -9.8841124, 10.2844543
1: -3.9131050, 3.4102268, -5.7598782, 4.8238630, -8.7369680, 9.1701050
2: -4.9935207, 3.4527514, -7.3339195, 4.8094854, -9.8030062, 10.7866688
3: -5.3967738, 2.8288891, -7.8829160, 3.9413977, -9.3381710, 10.7118053
4: -5.3863025, 3.7594538, -7.7990632, 5.3482261, -10.7345266, 11.5585175
5: -4.5990105, 3.8344724, -6.6346936, 5.2755542, -9.8745632, 10.4691648
6: -4.1194143, 4.2569590, -5.9102049, 5.9731164, -10.0925312, 10.1671638
7: -4.5788836, 4.4864898, -6.6444540, 6.4782176, -11.0571012, 11.1309433
8: -6.7199435, 3.0706205, -9.6615982, 4.1045771, -10.8245201, 12.7322187
9: -4.0594816, 4.0996399, -5.8306189, 5.8863440, -9.9458256, 9.9302578

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205560, upper bound: 12.3206368
time: 4.24 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205560, upper bound: 12.3208047
time: 3.50 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2.8311906, 2.3220236, -6.9278307, 5.6534505, -8.4846411, 9.2498541
1: -2.2613404, 2.1600585, -6.1198926, 5.1066923, -7.3680325, 8.2799511
2: -2.9074759, 2.2451537, -7.7954216, 5.0880952, -7.9955711, 10.0405750
3: -3.1814883, 1.8378270, -8.3744669, 4.1653776, -7.3468657, 10.2122936
4: -3.2304306, 2.3435440, -8.2662811, 5.6658688, -8.8962994, 10.6098251
5: -2.7489042, 2.5289159, -7.0471029, 5.5990324, -8.3479366, 9.5760193
6: -2.5099192, 2.7699656, -6.2874317, 6.3408647, -8.8507843, 9.0573978
7: -2.7814639, 2.7111759, -7.0439754, 6.8703451, -9.6518087, 9.7551498
8: -4.0767484, 2.2300501, -10.2549744, 4.3522644, -8.4290123, 12.4850245
9: -2.4844346, 2.5108836, -6.1865401, 6.2451653, -8.7296000, 8.6974239

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205223, upper bound: 12.3204203
time: 5.46 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206044, upper bound: 12.3205260
time: 6.40 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.5567398, 3.7537115, -7.7105088, 6.2758937, -10.8326311, 11.4642200
1: -3.9131050, 3.4102268, -6.8419647, 5.6589756, -9.5720806, 10.2521915
2: -4.9935207, 3.4527514, -8.7103539, 5.6195402, -10.6130600, 12.1631031
3: -5.3967738, 2.8288891, -9.3404226, 4.6019154, -9.9986897, 12.1693115
4: -5.3863025, 3.7594538, -9.2030525, 6.2869210, -11.6732216, 12.9625053
5: -4.5990105, 3.8344724, -7.8467627, 6.2111130, -10.8101225, 11.6812353
6: -4.1194143, 4.2569590, -6.9894147, 7.0163298, -11.1357441, 11.2463732
7: -4.5788836, 4.4864898, -7.8471293, 7.6470318, -12.2259140, 12.3336191
8: -6.7199435, 3.0706205, -11.4015112, 4.7855611, -11.5055008, 14.4721317
9: -4.0594816, 4.0996399, -6.8802962, 6.9447989, -11.0042782, 10.9799347

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205141, upper bound: 12.3205966
time: 3.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205141, upper bound: 12.3207888
time: 3.91 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.8669133, 3.1989429, -5.7768011, 4.7242775, -8.5911903, 8.9757442
1: -3.2499230, 2.9173281, -5.0637140, 4.2883568, -7.5382795, 7.9810419
2: -4.1628628, 2.9817739, -6.4475064, 4.2951202, -8.4579830, 9.4292793
3: -4.5234685, 2.4348857, -6.9483709, 3.5184388, -8.0419073, 9.3832560
4: -4.5215826, 3.2037556, -6.8928194, 4.7463875, -9.2679701, 10.0965748
5: -3.8708894, 3.3401735, -5.8592215, 4.7174950, -8.5883837, 9.1993952
6: -3.5120084, 3.7009213, -5.2313027, 5.3176832, -8.8296909, 8.9322243
7: -3.8463855, 3.7847154, -5.8675890, 5.7272558, -9.5736408, 9.6523018
8: -5.6910677, 2.7353666, -8.5508451, 3.6901088, -9.3811760, 11.2862110
9: -3.4377427, 3.4709694, -5.1589427, 5.2090325, -8.6467743, 8.6299105

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205110, upper bound: 12.3204364
time: 6.20 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205921, upper bound: 12.3205212
time: 5.08 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.6790013, 4.6648865, -6.5307431, 5.3273740, -11.0063753, 11.1956282
1: -4.9497828, 4.2107515, -5.7598782, 4.8238630, -9.7736454, 9.9706297
2: -6.3152680, 4.2339687, -7.3339195, 4.8094854, -11.1247520, 11.5678873
3: -6.7977176, 3.4640009, -7.8829160, 3.9413977, -10.7391148, 11.3469162
4: -6.7379527, 4.6596866, -7.7990632, 5.3482261, -12.0861769, 12.4587498
5: -5.7692528, 4.6893191, -6.6346936, 5.2755542, -11.0448055, 11.3240128
6: -5.1567168, 5.2623086, -5.9102049, 5.9731164, -11.1298332, 11.1725140
7: -5.7352281, 5.6052494, -6.6444540, 6.4782176, -12.2134457, 12.2497015
8: -8.3961763, 3.7126670, -9.6615982, 4.1045771, -12.5007534, 13.3742657
9: -5.0668244, 5.1196637, -5.8306189, 5.8863440, -10.9531689, 10.9502831

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205346, upper bound: 12.3206134
time: 3.45 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205346, upper bound: 12.3207928
time: 4.01 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.8669133, 3.1989429, -6.9278307, 5.6534505, -9.5203638, 10.1267719
1: -3.2499230, 2.9173281, -6.1198926, 5.1066923, -8.3566151, 9.0372200
2: -4.1628628, 2.9817739, -7.7954216, 5.0880952, -9.2509575, 10.7771940
3: -4.5234685, 2.4348857, -8.3744669, 4.1653776, -8.6888466, 10.8093529
4: -4.5215826, 3.2037556, -8.2662811, 5.6658688, -10.1874514, 11.4700365
5: -3.8708894, 3.3401735, -7.0471029, 5.5990324, -9.4699221, 10.3872766
6: -3.5120084, 3.7009213, -6.2874317, 6.3408647, -9.8528719, 9.9883528
7: -3.8463855, 3.7847154, -7.0439754, 6.8703451, -10.7167301, 10.8286905
8: -5.6910677, 2.7353666, -10.2549744, 4.3522644, -10.0433321, 12.9903402
9: -3.4377427, 3.4709694, -6.1865401, 6.2451653, -9.6829081, 9.6575069

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205211, upper bound: 12.3204444
time: 5.42 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206061, upper bound: 12.3205329
time: 3.81 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.6790013, 4.6648865, -7.7105088, 6.2758937, -11.9548941, 12.3753948
1: -4.9497828, 4.2107515, -6.8419647, 5.6589756, -10.6087580, 11.0527163
2: -6.3152680, 4.2339687, -8.7103539, 5.6195402, -11.9348059, 12.9443226
3: -6.7977176, 3.4640009, -9.3404226, 4.6019154, -11.3996334, 12.8044233
4: -6.7379527, 4.6596866, -9.2030525, 6.2869210, -13.0248737, 13.8627377
5: -5.7692528, 4.6893191, -7.8467627, 6.2111130, -11.9803658, 12.5360813
6: -5.1567168, 5.2623086, -6.9894147, 7.0163298, -12.1730452, 12.2517233
7: -5.7352281, 5.6052494, -7.8471293, 7.6470318, -13.3822584, 13.4523773
8: -8.3961763, 3.7126670, -11.4015112, 4.7855611, -13.1817341, 15.1141758
9: -5.0668244, 5.1196637, -6.8802962, 6.9447989, -12.0116234, 11.9999599

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205248, upper bound: 12.3206115
time: 4.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205248, upper bound: 12.3207953
time: 2.71 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.2867603, 3.5235767, -4.2766476, 3.5239797, -7.8107400, 7.8002238
1: -3.6725671, 3.2238162, -3.6465240, 3.2078359, -6.8804030, 6.8703403
2: -4.6826224, 3.2657063, -4.6572561, 3.2555492, -7.9381714, 7.9229622
3: -5.0824800, 2.6739926, -5.0397677, 2.6677334, -7.7502136, 7.7137604
4: -5.0763526, 3.5511825, -5.0384455, 3.5317862, -8.6081390, 8.5896273
5: -4.3027420, 3.6099644, -4.3007135, 3.6246099, -7.9273520, 7.9106779
6: -3.8814323, 4.0272188, -3.8607235, 4.0103245, -7.8917570, 7.8879423
7: -4.3121748, 4.2326927, -4.2816057, 4.2018089, -8.5139837, 8.5142984
8: -6.3324480, 2.8846362, -6.2959847, 2.9197886, -9.2522354, 9.1806211
9: -3.8209913, 3.8544624, -3.8042140, 3.8403230, -7.6613140, 7.6586761

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205557, upper bound: 12.3204380
time: 5.09 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206284, upper bound: 12.3205264
time: 3.99 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.0695972, 4.9574566, -4.9937305, 4.1106253, -10.1802206, 9.9511871
1: -5.3332987, 4.4959083, -4.3249941, 3.7235610, -9.0568590, 8.8209019
2: -6.7900853, 4.4941902, -5.5168214, 3.7579212, -10.5480061, 10.0110111
3: -7.3103271, 3.6821315, -5.9486189, 3.0790124, -10.3893394, 9.6307507
4: -7.2438035, 4.9788942, -5.9233742, 4.1134143, -11.3572178, 10.9022665
5: -6.1613383, 4.9326992, -5.0578213, 4.1587343, -10.3200722, 9.9905205
6: -5.4942732, 5.5706878, -4.5201321, 4.6408486, -10.1351204, 10.0908194
7: -6.1681499, 6.0177870, -5.0393720, 4.9274111, -11.0955582, 11.0571594
8: -8.9791651, 3.8501067, -7.3781786, 3.3124800, -12.2916431, 11.2282848
9: -5.4188428, 5.4709911, -4.4551797, 4.5017738, -9.9206161, 9.9261703

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205685, upper bound: 12.3206374
time: 4.06 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205685, upper bound: 12.3207986
time: 3.67 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.2867603, 3.5235767, -5.7613816, 4.7128248, -8.9995852, 9.2849579
1: -3.6725671, 3.2238162, -5.0501809, 4.2771606, -7.9497280, 8.2739964
2: -4.6826224, 3.2657063, -6.4303174, 4.2851810, -8.9678040, 9.6960239
3: -5.0824800, 2.6739926, -6.9300704, 3.5097864, -8.5922661, 9.6040630
4: -5.0763526, 3.5511825, -6.8751206, 4.7326136, -9.8089657, 10.4263020
5: -4.3027420, 3.6099644, -5.8444037, 4.7064838, -9.0092258, 9.4543686
6: -3.8814323, 4.0272188, -5.2155557, 5.3052125, -9.1866446, 9.2427750
7: -4.3121748, 4.2326927, -5.8515377, 5.7126050, -10.0247803, 10.0842304
8: -6.3324480, 2.8846362, -8.5294104, 3.6825259, -10.0149727, 11.4140463
9: -3.8209913, 3.8544624, -5.1451683, 5.1959729, -9.0169640, 8.9996300

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205446, upper bound: 12.3204316
time: 2.75 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206229, upper bound: 12.3205242
time: 4.06 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.0695972, 4.9574566, -6.4879503, 5.2926636, -11.3622599, 11.4454060
1: -5.3332987, 4.4959083, -5.7203236, 4.7929153, -10.1262131, 10.2162304
2: -6.7900853, 4.4941902, -7.2848334, 4.7814498, -11.5715351, 11.7790222
3: -7.3103271, 3.6821315, -7.8311834, 3.9167318, -11.2270584, 11.5133133
4: -7.2438035, 4.9788942, -7.7490311, 5.3102360, -12.5540380, 12.7279224
5: -6.1613383, 4.9326992, -6.5929055, 5.2412081, -11.4025459, 11.5256042
6: -5.4942732, 5.5706878, -5.8677812, 5.9376740, -11.4319458, 11.4384689
7: -6.1681499, 6.0177870, -6.5998178, 6.4354320, -12.6035786, 12.6176052
8: -8.9791651, 3.8501067, -9.5999622, 4.0830383, -13.0622025, 13.4500694
9: -5.4188428, 5.4709911, -5.7914481, 5.8493953, -11.2682381, 11.2624388

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205541, upper bound: 12.3206325
time: 3.60 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205541, upper bound: 12.3207971
time: 4.06 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.3588667, 4.4036984, -4.2766476, 3.5239797, -8.8828430, 8.6803455
1: -4.6708040, 3.9951825, -3.6465240, 3.2078359, -7.8786402, 7.6417065
2: -5.9554720, 4.0208092, -4.6572561, 3.2555492, -9.2110214, 8.6780653
3: -6.4340324, 3.2864108, -5.0397677, 2.6677334, -9.1017656, 8.3261776
4: -6.3789105, 4.4190760, -5.0384455, 3.5317862, -9.9106951, 9.4575205
5: -5.4319439, 4.4389167, -4.3007135, 3.6246099, -9.0565538, 8.7396297
6: -4.8847308, 4.9948697, -3.8607235, 4.0103245, -8.8950539, 8.8555927
7: -5.4278159, 5.3099856, -4.2816057, 4.2018089, -9.6296253, 9.5915909
8: -7.9497743, 3.5019846, -6.2959847, 2.9197886, -10.8695631, 9.7979698
9: -4.7925735, 4.8393064, -3.8042140, 3.8403230, -8.6328964, 8.6435204

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205119, upper bound: 12.3204038
time: 3.78 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205897, upper bound: 12.3204876
time: 3.94 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.2644320, 5.9201159, -4.9937305, 4.1106253, -11.3750572, 10.9138470
1: -6.4302292, 5.3439670, -4.3249941, 3.7235610, -10.1537905, 9.6689606
2: -8.1871872, 5.3162642, -5.5168214, 3.7579212, -11.9451056, 10.8330860
3: -8.7894611, 4.3526797, -5.9486189, 3.0790124, -11.8684731, 10.3012981
4: -8.6691399, 5.9318686, -5.9233742, 4.1134143, -12.7825546, 11.8552408
5: -7.3920546, 5.8619661, -5.0578213, 4.1587343, -11.5507889, 10.9197855
6: -6.5888543, 6.6303730, -4.5201321, 4.6408486, -11.2297020, 11.1505041
7: -7.3885794, 7.2038231, -5.0393720, 4.9274111, -12.3159904, 12.2431946
8: -10.7463255, 4.5347948, -7.3781786, 3.3124800, -14.0588036, 11.9129734
9: -6.4843087, 6.5450959, -4.4551797, 4.5017738, -10.9860821, 11.0002756

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205431, upper bound: 12.3206113
time: 4.72 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205431, upper bound: 12.3207852
time: 3.77 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.3588667, 4.4036984, -5.7613816, 4.7128248, -10.0716906, 10.1650801
1: -4.6708040, 3.9951825, -5.0501809, 4.2771606, -8.9479647, 9.0453634
2: -5.9554720, 4.0208092, -6.4303174, 4.2851810, -10.2406530, 10.4511261
3: -6.4340324, 3.2864108, -6.9300704, 3.5097864, -9.9438190, 10.2164783
4: -6.3789105, 4.4190760, -6.8751206, 4.7326136, -11.1115208, 11.2941952
5: -5.4319439, 4.4389167, -5.8444037, 4.7064838, -10.1384277, 10.2833204
6: -4.8847308, 4.9948697, -5.2155557, 5.3052125, -10.1899433, 10.2104254
7: -5.4278159, 5.3099856, -5.8515377, 5.7126050, -11.1404209, 11.1615238
8: -7.9497743, 3.5019846, -8.5294104, 3.6825259, -11.6323004, 12.0313950
9: -4.7925735, 4.8393064, -5.1451683, 5.1959729, -9.9885464, 9.9844742

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204981, upper bound: 12.3203965
time: 4.14 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205823, upper bound: 12.3204851
time: 3.84 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.2644320, 5.9201159, -6.4879503, 5.2926636, -12.5570955, 12.4080658
1: -6.4302292, 5.3439670, -5.7203236, 4.7929153, -11.2231436, 11.0642910
2: -8.1871872, 5.3162642, -7.2848334, 4.7814498, -12.9686346, 12.6010971
3: -8.7894611, 4.3526797, -7.8311834, 3.9167318, -12.7061930, 12.1838627
4: -8.6691399, 5.9318686, -7.7490311, 5.3102360, -13.9793749, 13.6808987
5: -7.3920546, 5.8619661, -6.5929055, 5.2412081, -12.6332626, 12.4548721
6: -6.5888543, 6.6303730, -5.8677812, 5.9376740, -12.5265255, 12.4981537
7: -7.3885794, 7.2038231, -6.5998178, 6.4354320, -13.8240108, 13.8036404
8: -10.7463255, 4.5347948, -9.5999622, 4.0830383, -14.8293638, 14.1347570
9: -6.4843087, 6.5450959, -5.7914481, 5.8493953, -12.3337040, 12.3365421

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205298, upper bound: 12.3206049
time: 4.59 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205298, upper bound: 12.3207829
time: 3.05 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 6.71 + 595.50 = 602.21 seconds
