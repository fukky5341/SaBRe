## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 11.027958876


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610)
1: (-5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247)
2: (-6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207)
3: (-7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554)
4: (-7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166)
5: (-6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819)
6: (-6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685)
7: (-7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842)
8: (-7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869)
9: (-6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.74 + 5.53 = 6.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -11.1393523, upper bound: 11.1393524

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1258847, upper bound: 11.1249726
time: 11.71 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1392413, upper bound: 11.1392413
time: 2.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 14.62 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 14.62
Output dim: 7, lower bound: -11.1258847, upper bound: 11.1249726
NS_B2, status: Status.UNKNOWN, split count: 1, time: 14.62
Output dim: 7, lower bound: -11.1392413, upper bound: 11.1392413

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -6.5408664, 5.5315504, -6.8031778, 5.7365303, -12.2773952, 12.3347282
1: -5.2291765, 4.7948642, -5.4268703, 4.9493256, -10.1785011, 10.2217350
2: -6.7434249, 4.4685254, -6.9864039, 4.5101366, -11.2535610, 11.4549294
3: -7.5769348, 3.9080787, -7.9464793, 3.9219804, -11.4989147, 11.8545570
4: -7.3340139, 5.7357998, -7.6412487, 6.0155678, -13.3495817, 13.3770485
5: -6.3652673, 5.0766630, -6.5650902, 5.2264585, -11.5917244, 11.6417522
6: -5.9306264, 6.3051825, -6.1009660, 6.6026506, -12.5332747, 12.4061489
7: -7.3453012, 4.7119579, -7.6735902, 4.6290693, -11.9743690, 12.3855476
8: -7.3520002, 5.3517675, -7.5839128, 5.5273619, -12.8793621, 12.9356804
9: -5.9191351, 6.0162024, -6.1305461, 6.2312937, -12.1504288, 12.1467485

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 188

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1225680, upper bound: 11.1213891
time: 3.77 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1234755, upper bound: 11.1220705
time: 4.41 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -6.7478456, 5.7016158, -6.6649265, 5.6336174, -12.3814621, 12.3665400
1: -5.4021134, 4.9384117, -5.3328838, 4.8809633, -10.2830772, 10.2712955
2: -6.9570780, 4.6066432, -6.8713760, 4.5503244, -11.5074024, 11.4780197
3: -7.8119092, 4.0386467, -7.7183523, 3.9860766, -11.7979860, 11.7569981
4: -7.5572267, 5.8959913, -7.4682083, 5.8314939, -13.3887196, 13.3641996
5: -6.5690393, 5.2315445, -6.4871206, 5.1695743, -11.7386122, 11.7186651
6: -6.1139588, 6.4829106, -6.0401793, 6.4118648, -12.5258236, 12.5230875
7: -7.5546770, 4.8859076, -7.4717345, 4.8148556, -12.3695316, 12.3576422
8: -7.5906458, 5.5113425, -7.4947567, 5.4467425, -13.0373878, 13.0060978
9: -6.1002841, 6.2026968, -6.0273323, 6.1278629, -12.2281466, 12.2300291

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1324266, upper bound: 11.1324713
time: 4.55 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1309096, upper bound: 11.1309096
time: 2.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 8.04 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 8.04
Output dim: 7, lower bound: -11.1225680, upper bound: 11.1213891
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 8.04
Output dim: 7, lower bound: -11.1234755, upper bound: 11.1220705
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 8.04
Output dim: 7, lower bound: -11.1324266, upper bound: 11.1324713
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 8.04
Output dim: 7, lower bound: -11.1309096, upper bound: 11.1309096

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -4.5744228, 3.8640525, -6.2507482, 5.2790480, -9.8534708, 10.1148005
1: -3.5252075, 3.4119380, -4.9510698, 4.5656743, -8.0908813, 8.3630075
2: -4.6719408, 3.1021495, -6.4114866, 4.1304226, -8.8023634, 9.5136356
3: -5.2895856, 2.7272158, -7.3036981, 3.6017828, -8.8913689, 10.0309124
4: -5.1002498, 4.1211710, -7.0220742, 5.5625143, -10.6627636, 11.1432457
5: -4.3063946, 3.5411646, -5.9953432, 4.7903504, -9.0967445, 9.5365076
6: -4.1372309, 4.5380106, -5.6040087, 6.1073976, -10.2446289, 10.1420193
7: -5.2028780, 2.9970264, -7.0791755, 4.1575947, -9.3604708, 10.0762024
8: -5.1302977, 3.7843778, -6.9752588, 5.0848684, -10.2151661, 10.7596369
9: -4.1421614, 4.1759424, -5.6395855, 5.7202034, -9.8623648, 9.8155270

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1225066, upper bound: 11.1212255
time: 4.31 seconds

## Relational analysis of NS_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1225066, upper bound: 11.1213891
time: 3.77 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -5.5281343, 4.6863813, -6.6220293, 5.5887957, -11.1169281, 11.3084106
1: -4.3680534, 4.0923243, -5.2712431, 4.8230157, -9.1910686, 9.3635645
2: -5.7016325, 3.7813463, -6.7997065, 4.3863392, -10.0879717, 10.5810509
3: -6.4184365, 3.3160992, -7.7338290, 3.8175335, -10.2359695, 11.0499287
4: -6.2030582, 4.9158473, -7.4371614, 5.8665614, -12.0696201, 12.3530064
5: -5.3431625, 4.2918596, -6.3804317, 5.0872831, -10.4304457, 10.6722908
6: -5.0285292, 5.4121289, -5.9373446, 6.4411411, -11.4696693, 11.3494740
7: -6.2777100, 3.8834634, -7.4797716, 4.4821267, -10.7598362, 11.3632355
8: -6.2183871, 4.5599732, -7.3853812, 5.3836389, -11.6020241, 11.9453545
9: -5.0229177, 5.0965190, -5.9711208, 6.0651298, -11.0880470, 11.0676384

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1229936, upper bound: 11.1216162
time: 3.88 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1229936, upper bound: 11.1220705
time: 3.74 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -6.4082499, 5.4196887, -6.6649265, 5.6336174, -12.0418663, 12.0846148
1: -5.1161637, 4.7054176, -5.3328838, 4.8809633, -9.9971256, 10.0383015
2: -6.6072059, 4.3798723, -6.8713760, 4.5503244, -11.1575298, 11.2512484
3: -7.4170475, 3.8383527, -7.7183523, 3.9860766, -11.4031239, 11.5567055
4: -7.1759787, 5.6193500, -7.4682083, 5.8314939, -13.0074730, 13.0875587
5: -6.2298455, 4.9709430, -6.4871206, 5.1695743, -11.3994198, 11.4580631
6: -5.8129530, 6.1855145, -6.0401793, 6.4118648, -12.2248173, 12.2256937
7: -7.2034159, 4.6160998, -7.4717345, 4.8148556, -12.0182695, 12.0878334
8: -7.2060909, 5.2446194, -7.4947567, 5.4467425, -12.6528330, 12.7393761
9: -5.7991829, 5.8931599, -6.0273323, 6.1278629, -11.9270458, 11.9204922

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1276412, upper bound: 11.1272514
time: 3.79 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1308100, upper bound: 11.1308075
time: 3.97 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -5.2869511, 4.4547110, -6.2145462, 5.2596622, -10.5466137, 10.6692562
1: -4.1359158, 3.9093082, -4.9531746, 4.5724859, -8.7084017, 8.8624821
2: -5.4387102, 3.5314331, -6.4084034, 4.2483110, -9.6870213, 9.9398365
3: -6.1459117, 3.1254158, -7.1994381, 3.7177644, -9.8636751, 10.3248539
4: -5.8773479, 4.7190485, -6.9666481, 5.4677544, -11.3451023, 11.6856956
5: -5.0242925, 4.0595641, -6.0376511, 4.8225303, -9.8468227, 10.0972157
6: -4.7705789, 5.2180290, -5.6410384, 6.0192318, -10.7898102, 10.8590641
7: -6.0054502, 3.5436971, -7.0066876, 4.4514723, -10.4569225, 10.5503845
8: -5.9268217, 4.3456569, -6.9832215, 5.0948181, -11.0216398, 11.3288784
9: -4.7787194, 4.8431206, -5.6283226, 5.7189217, -10.4976397, 10.4714432

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1262215, upper bound: 11.1258713
time: 3.13 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1292490, upper bound: 11.1292490
time: 3.47 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 7.39 seconds
NS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 7.39
Output dim: 7, lower bound: -11.1225066, upper bound: 11.1212255
NS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 7.39
Output dim: 7, lower bound: -11.1225066, upper bound: 11.1213891
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 7.39
Output dim: 7, lower bound: -11.1229936, upper bound: 11.1216162
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 7.39
Output dim: 7, lower bound: -11.1229936, upper bound: 11.1220705
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 7.39
Output dim: 7, lower bound: -11.1276412, upper bound: 11.1272514
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 7.39
Output dim: 7, lower bound: -11.1308100, upper bound: 11.1308075
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 7.39
Output dim: 7, lower bound: -11.1262215, upper bound: 11.1258713
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 7.39
Output dim: 7, lower bound: -11.1292490, upper bound: 11.1292490

## BFS NS instance: NS_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.5744228, 3.8640525, -5.1864638, 4.3578529, -8.9322758, 9.0505161
1: -3.5252075, 3.4119380, -4.0422354, 3.8235800, -7.3487873, 7.4541731
2: -4.6719408, 3.1021495, -5.2643023, 3.3974533, -8.0693941, 8.3664513
3: -5.2895856, 2.7272158, -6.0598745, 2.9941173, -8.2837029, 8.7870903
4: -5.1002498, 4.1211710, -5.7920265, 4.6691141, -9.7693634, 9.9131975
5: -4.3063946, 3.5411646, -4.8655477, 3.9480324, -8.2544270, 8.4067125
6: -4.1372309, 4.5380106, -4.6559863, 5.1183386, -9.2555695, 9.1939964
7: -5.2028780, 2.9970264, -5.8829145, 3.2477770, -8.4506550, 8.8799410
8: -5.1302977, 3.7843778, -5.7833338, 4.2279506, -9.3582478, 9.5677118
9: -4.1421614, 4.1759424, -4.6677103, 4.6918378, -8.8339996, 8.8436527

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 107

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0913966, upper bound: 11.0317570
time: 4.31 seconds

## Relational analysis of NS_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1225066, upper bound: 11.1212255
time: 3.60 seconds

## BFS NS instance: NS_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.5744228, 3.8640525, -5.9462008, 5.0301385, -9.6045609, 9.8102522
1: -3.5252075, 3.4119380, -4.6917210, 4.3515415, -7.8767490, 8.1036587
2: -4.6719408, 3.1021495, -6.0979252, 3.9255395, -8.5974808, 9.2000751
3: -5.2895856, 2.7272158, -6.9426851, 3.4317157, -8.7213011, 9.6698980
4: -5.1002498, 4.1211710, -6.6729393, 5.3072672, -10.4075165, 10.7941103
5: -4.3063946, 3.5411646, -5.6887264, 4.5628657, -8.8692608, 9.2298908
6: -4.1372309, 4.5380106, -5.3339663, 5.8307419, -9.9679728, 9.8719769
7: -5.2028780, 2.9970264, -6.7502947, 3.9317126, -9.1345882, 9.7473211
8: -5.1302977, 3.7843778, -6.6413298, 4.8465104, -9.9768085, 10.4257050
9: -4.1421614, 4.1759424, -5.3739576, 5.4398603, -9.5820217, 9.5499001

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 146

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of NS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_B1_A1_B2_B1

### Relational analysis result of NS_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1197700, upper bound: 11.1189134
time: 3.48 seconds

## Relational analysis of NS_B1_A1_B2_B2

### Relational analysis result of NS_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1164038, upper bound: 11.0073628
time: 3.96 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.5281343, 4.6863813, -5.1864638, 4.3578529, -9.8859854, 9.8728447
1: -4.3680534, 4.0923243, -4.0422354, 3.8235800, -8.1916304, 8.1345596
2: -5.7016325, 3.7813463, -5.2643023, 3.3974533, -9.0990858, 9.0456486
3: -6.4184365, 3.3160992, -6.0598745, 2.9941173, -9.4125538, 9.3759737
4: -6.2030582, 4.9158473, -5.7920265, 4.6691141, -10.8721724, 10.7078724
5: -5.3431625, 4.2918596, -4.8655477, 3.9480324, -9.2911949, 9.1574068
6: -5.0285292, 5.4121289, -4.6559863, 5.1183386, -10.1468668, 10.0681152
7: -6.2777100, 3.8834634, -5.8829145, 3.2477770, -9.5254869, 9.7663774
8: -6.2183871, 4.5599732, -5.7833338, 4.2279506, -10.4463367, 10.3433075
9: -5.0229177, 5.0965190, -4.6677103, 4.6918378, -9.7147551, 9.7642288

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1193433, upper bound: 11.0869943
time: 3.24 seconds

## Relational analysis of NS_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1193780, upper bound: 11.0869746
time: 3.29 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.5281343, 4.6863813, -5.9462008, 5.0301385, -10.5582705, 10.6325817
1: -4.3680534, 4.0923243, -4.6917210, 4.3515415, -8.7195930, 8.7840433
2: -5.7016325, 3.7813463, -6.0979252, 3.9255395, -9.6271725, 9.8792706
3: -6.4184365, 3.3160992, -6.9426851, 3.4317157, -9.8501520, 10.2587843
4: -6.2030582, 4.9158473, -6.6729393, 5.3072672, -11.5103254, 11.5887861
5: -5.3431625, 4.2918596, -5.6887264, 4.5628657, -9.9060287, 9.9805851
6: -5.0285292, 5.4121289, -5.3339663, 5.8307419, -10.8592710, 10.7460938
7: -6.2777100, 3.8834634, -6.7502947, 3.9317126, -10.2094231, 10.6337585
8: -6.2183871, 4.5599732, -6.6413298, 4.8465104, -11.0648975, 11.2013035
9: -5.0229177, 5.0965190, -5.3739576, 5.4398603, -10.4627762, 10.4704762

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 62

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1074207, upper bound: 11.0375426
time: 3.23 seconds

## Relational analysis of NS_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1229936, upper bound: 11.1220705
time: 17.18 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.7361183, 4.8533425, -4.6506619, 3.9273903, -9.6635084, 9.5040045
1: -4.5460401, 4.2412977, -3.5918355, 3.4686661, -8.0147047, 7.8331332
2: -5.9129109, 3.9205291, -4.7561903, 3.1554313, -9.0683422, 8.6767197
3: -6.6575689, 3.4389141, -5.3846388, 2.7772379, -9.4348049, 8.8235531
4: -6.4318986, 5.0808926, -5.1896172, 4.1842041, -10.6161022, 10.2705078
5: -5.5463138, 4.4385128, -4.3880510, 3.5961809, -9.1424942, 8.8265638
6: -5.2169018, 5.5958843, -4.2093425, 4.6101899, -9.8270912, 9.8052273
7: -6.4992638, 4.0417275, -5.2906771, 3.0606380, -9.5599022, 9.3324051
8: -6.4490533, 4.7176037, -5.2177114, 3.8465910, -10.2956419, 9.9353151
9: -5.1999531, 5.2824326, -4.2104707, 4.2491322, -9.4490852, 9.4929028

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 62

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1271331, upper bound: 11.1269069
time: 3.36 seconds

## Relational analysis of NS_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1271331, upper bound: 11.1272514
time: 3.80 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.1834564, 5.2336950, -5.6321340, 4.7739940, -10.9574490, 10.8658295
1: -4.9278798, 4.5514851, -4.4582376, 4.1670780, -9.0949554, 9.0097227
2: -6.3775539, 4.2291851, -5.8096819, 3.8521299, -10.2296839, 10.0388670
3: -7.1649151, 3.7052929, -6.5404949, 3.3813059, -10.5462208, 10.2457876
4: -6.9289365, 5.4374080, -6.3185225, 4.9957252, -11.9246616, 11.7559299
5: -6.0066900, 4.7957606, -5.4498758, 4.3682470, -10.3749371, 10.2456360
6: -5.6159530, 5.9896526, -5.1249552, 5.5032926, -11.1192446, 11.1146078
7: -6.9719386, 4.4332876, -6.3899922, 3.9725924, -10.9445305, 10.8232803
8: -6.9527240, 5.0687447, -6.3383703, 4.6399903, -11.5927143, 11.4071150
9: -5.6000385, 5.6910501, -5.1139555, 5.1916995, -10.7917385, 10.8050060

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1283849, upper bound: 11.1286259
time: 3.26 seconds

## Relational analysis of NS_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1283849, upper bound: 11.1308074
time: 3.28 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.8096061, 4.0427065, -4.3444147, 3.6630158, -8.4726219, 8.3871212
1: -3.7159479, 3.5654666, -3.3237481, 3.2465346, -6.9624825, 6.8892145
2: -4.9107666, 3.2039785, -4.4240656, 2.9369316, -7.8476954, 7.6280441
3: -5.5733905, 2.8373909, -5.0061455, 2.5933790, -8.1667690, 7.8435364
4: -5.3228307, 4.3135777, -4.8212271, 3.9263148, -9.2491455, 9.1348038
5: -4.5113540, 3.6820767, -4.0586762, 3.3646750, -7.8760290, 7.7407513
6: -4.3267069, 4.7709532, -3.9242003, 4.3213425, -8.6480494, 8.6951532
7: -5.4571705, 3.1074793, -4.9383984, 2.8050406, -8.2622108, 8.0458775
8: -5.3873525, 3.9582891, -4.8682251, 3.6020455, -8.9893970, 8.8265142
9: -4.3369613, 4.3820362, -3.9307845, 3.9556148, -8.2925749, 8.3128204

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 188

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1250220, upper bound: 11.1250220
time: 2.15 seconds

## Relational analysis of NS_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1250220, upper bound: 11.1258713
time: 2.14 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.1298699, 4.3214297, -5.2550688, 4.4531546, -9.5830231, 9.5764980
1: -3.9969282, 3.7944558, -4.1278057, 3.8999507, -7.8968782, 7.9222612
2: -5.2671633, 3.4259973, -5.4177375, 3.5898154, -8.8569775, 8.8437347
3: -5.9567332, 3.0314963, -6.0902853, 3.1582687, -9.1150017, 9.1217804
4: -5.6948948, 4.5846710, -5.8840542, 4.6870537, -10.3819485, 10.4687252
5: -4.8580666, 3.9385295, -5.0547428, 4.0843000, -8.9423656, 8.9932709
6: -4.6244974, 5.0709348, -4.7750130, 5.1669002, -9.7913961, 9.8459473
7: -5.8256841, 3.4092956, -5.9776363, 3.6566114, -9.4822960, 9.3869324
8: -5.7508888, 4.2178221, -5.9084744, 4.3449755, -10.0958614, 10.1262960
9: -4.6351109, 4.6919498, -4.7770643, 4.8386016, -9.4737110, 9.4690142

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 148

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1258713, upper bound: 11.1262215
time: 3.08 seconds

## Relational analysis of NS_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1258713, upper bound: 11.1292490
time: 3.47 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 12.84 seconds
NS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 12.84
Output dim: 7, lower bound: -11.0913966, upper bound: 11.0317570
NS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 12.84
Output dim: 7, lower bound: -11.1225066, upper bound: 11.1212255
NS_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 12.84
Output dim: 7, lower bound: -11.1197700, upper bound: 11.1189134
NS_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 12.84
Output dim: 7, lower bound: -11.1164038, upper bound: 11.0073628
NS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 12.84
Output dim: 7, lower bound: -11.1193433, upper bound: 11.0869943
NS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 12.84
Output dim: 7, lower bound: -11.1193780, upper bound: 11.0869746
NS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 12.84
Output dim: 7, lower bound: -11.1074207, upper bound: 11.0375426
NS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 12.84
Output dim: 7, lower bound: -11.1229936, upper bound: 11.1220705
NS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 12.84
Output dim: 7, lower bound: -11.1271331, upper bound: 11.1269069
NS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 12.84
Output dim: 7, lower bound: -11.1271331, upper bound: 11.1272514
NS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 12.84
Output dim: 7, lower bound: -11.1283849, upper bound: 11.1286259
NS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 12.84
Output dim: 7, lower bound: -11.1283849, upper bound: 11.1308074
NS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 12.84
Output dim: 7, lower bound: -11.1250220, upper bound: 11.1250220
NS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 12.84
Output dim: 7, lower bound: -11.1250220, upper bound: 11.1258713
NS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 12.84
Output dim: 7, lower bound: -11.1258713, upper bound: 11.1262215
NS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 12.84
Output dim: 7, lower bound: -11.1258713, upper bound: 11.1292490

## BFS NS instance: NS_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.6957765, 3.9242623, -4.9910803, 4.1865163, -8.8822927, 8.9153423
1: -3.6004856, 3.4749079, -3.8721504, 3.6833088, -7.2837944, 7.3470583
2: -4.7387257, 3.0926251, -5.0500469, 3.2704675, -8.0091934, 8.1426716
3: -5.4384098, 2.7352300, -5.8131685, 2.8859425, -8.3243523, 8.5483990
4: -5.1805401, 4.2105713, -5.5563722, 4.4989576, -9.6794977, 9.7669430
5: -4.3308239, 3.5488086, -4.6531296, 3.7965858, -8.1274099, 8.2019386
6: -4.2085271, 4.6330347, -4.4796081, 4.9300327, -9.1385593, 9.1126423
7: -5.2811546, 2.8557394, -5.6546888, 3.0932393, -8.3743935, 8.5104284
8: -5.2385731, 3.8500896, -5.5587187, 4.0756321, -9.3142052, 9.4088078
9: -4.1920543, 4.2246346, -4.4857845, 4.5032077, -8.6952620, 8.7104187

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 195

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of NS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_B1_A1_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9076481, upper bound: 10.8803418
time: 2.89 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.8994532, upper bound: 10.8733505
time: 3.52 seconds

## BFS NS instance: NS_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.4072294, 3.7170155, -5.1864638, 4.3578529, -8.7650824, 8.9034796
1: -3.3769495, 3.2878823, -4.0422354, 3.8235800, -7.2005296, 7.3301177
2: -4.4849176, 2.9860685, -5.2643023, 3.3974533, -7.8823709, 8.2503710
3: -5.0772867, 2.6246707, -6.0598745, 2.9941173, -8.0714035, 8.6845455
4: -4.8955069, 3.9768665, -5.7920265, 4.6691141, -9.5646210, 9.7688932
5: -4.1209269, 3.4117231, -4.8655477, 3.9480324, -8.0689592, 8.2772713
6: -3.9807222, 4.3750391, -4.6559863, 5.1183386, -9.0990610, 9.0310249
7: -5.0045075, 2.8537393, -5.8829145, 3.2477770, -8.2522850, 8.7366543
8: -4.9366379, 3.6500447, -5.7833338, 4.2279506, -9.1645889, 9.4333782
9: -3.9865377, 4.0115771, -4.6677103, 4.6918378, -8.6783752, 8.6792870

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 195

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_B1_A1_B1_A2_A1

### Relational analysis result of NS_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1168698, upper bound: 11.0060424
time: 4.35 seconds

## Relational analysis of NS_B1_A1_B1_A2_A2

### Relational analysis result of NS_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1164730, upper bound: 11.0057913
time: 3.12 seconds

## BFS NS instance: NS_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -4.4772034, 3.7791736, -5.4494925, 4.6088858, -9.0860891, 9.2286663
1: -3.4399538, 3.3409681, -4.2731528, 4.0089622, -7.4489160, 7.6141210
2: -4.5657778, 3.0315027, -5.5659952, 3.5977566, -8.1635342, 8.5974979
3: -5.1708360, 2.6679556, -6.3565207, 3.1551204, -8.3259563, 9.0244761
4: -4.9837265, 4.0384283, -6.1090908, 4.8896503, -9.8733768, 10.1475191
5: -4.2014155, 3.4674623, -5.1782761, 4.1790099, -8.3804255, 8.6457386
6: -4.0468721, 4.4451342, -4.8949704, 5.3730335, -9.4199057, 9.3401051
7: -5.0889773, 2.9163446, -6.1977100, 3.5352116, -8.6241894, 9.1140547
8: -5.0190229, 3.7057309, -6.0882902, 4.4527435, -9.4717665, 9.7940216
9: -4.0533638, 4.0817270, -4.9323845, 4.9717121, -9.0250759, 9.0141115

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 146

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_B1_A1_B2_B1_A1

### Relational analysis result of NS_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1164038, upper bound: 11.0073628
time: 4.12 seconds

## Relational analysis of NS_B1_A1_B2_B1_A2

### Relational analysis result of NS_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1164038, upper bound: 11.0073628
time: 5.17 seconds

## BFS NS instance: NS_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -4.3786926, 3.6920071, -5.8768077, 4.9431505, -9.3218431, 9.5688152
1: -3.3529994, 3.2691402, -4.6315069, 4.3059874, -7.6589870, 7.9006472
2: -4.4580679, 2.9573157, -5.9966087, 3.8255706, -8.2836380, 8.9539242
3: -5.0510035, 2.6079183, -6.8961678, 3.3764992, -8.4275026, 9.5040855
4: -4.8642712, 3.9546821, -6.5799365, 5.2423944, -10.1066656, 10.5346184
5: -4.0929079, 3.3905435, -5.5736613, 4.4555941, -8.5485020, 8.9642048
6: -3.9554057, 4.3514347, -5.2616062, 5.7724109, -9.7278166, 9.6130409
7: -4.9735889, 2.8299286, -6.6583509, 3.7552495, -8.7288380, 9.4882793
8: -4.9065008, 3.6262867, -6.5595951, 4.7728205, -9.6793213, 10.1858816
9: -3.9623930, 3.9854851, -5.2905622, 5.3465266, -9.3089199, 9.2760468

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_B1_A1_B2_B2_A1

### Relational analysis result of NS_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1164038, upper bound: 11.0073627
time: 3.86 seconds

## Relational analysis of NS_B1_A1_B2_B2_A2

### Relational analysis result of NS_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1164038, upper bound: 11.0073627
time: 4.33 seconds

## BFS NS instance: NS_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.2643251, 3.6176028, -4.9442382, 4.1477203, -8.4120445, 8.5618410
1: -3.2729497, 3.1902621, -3.8339763, 3.6507661, -6.9237156, 7.0242386
2: -4.3465343, 2.9136219, -5.0002880, 3.2426953, -7.5892296, 7.9139099
3: -4.8863459, 2.5611954, -5.7538104, 2.8608918, -7.7472377, 8.3150063
4: -4.7385626, 3.8574886, -5.5037842, 4.4599276, -9.1984901, 9.3612728
5: -3.9810562, 3.3332303, -4.6010432, 3.7626524, -7.7437086, 7.9342718
6: -3.8573647, 4.2425499, -4.4400868, 4.8867931, -8.7441578, 8.6826363
7: -4.8548388, 2.7836242, -5.6022453, 3.0568135, -7.9116526, 8.3858700
8: -4.7798905, 3.5581045, -5.5068407, 4.0415030, -8.8213921, 9.0649452
9: -3.8749232, 3.8916588, -4.4435339, 4.4605536, -8.3354769, 8.3351927

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 107

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_B1_A2_B1_A1_A1

### Relational analysis result of NS_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -11.0135655, upper bound: 10.9362424
time: 3.98 seconds

## Relational analysis of NS_B1_A2_B1_A1_A2

### Relational analysis result of NS_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1193433, upper bound: 11.0869943
time: 3.41 seconds

## BFS NS instance: NS_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.8042078, 4.0689430, -4.8386889, 4.0582848, -8.8624926, 8.9076319
1: -3.7387209, 3.5740337, -3.7442331, 3.5751555, -7.3138762, 7.3182669
2: -4.9197574, 3.2469783, -4.8879538, 3.1801243, -8.0998821, 8.1349316
3: -5.5713234, 2.8543987, -5.6208000, 2.8043032, -8.3756266, 8.4751987
4: -5.3726263, 4.3245196, -5.3806238, 4.3700671, -9.7426929, 9.7051430
5: -4.5353093, 3.7222204, -4.4956808, 3.6870666, -8.2223759, 8.2179012
6: -4.3413472, 4.7579288, -4.3474541, 4.7861338, -9.1274805, 9.1053829
7: -5.4715266, 3.1585081, -5.4824762, 2.9912453, -8.4627724, 8.6409845
8: -5.3923268, 3.9723761, -5.3890414, 3.9604700, -9.3527946, 9.3614178
9: -4.3584714, 4.3918290, -4.3509970, 4.3637633, -8.7222347, 8.7428265

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_B1_A2_B1_A2_A1

### Relational analysis result of NS_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9597565, upper bound: 10.9017391
time: 3.94 seconds

## Relational analysis of NS_B1_A2_B1_A2_A2

### Relational analysis result of NS_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1193779, upper bound: 11.0869746
time: 3.78 seconds

## BFS NS instance: NS_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.7513757, 4.8261790, -5.7106733, 4.8304787, -10.5818539, 10.5368519
1: -4.5284433, 4.2284870, -4.4902420, 4.1892529, -8.7176952, 8.7187271
2: -5.8862128, 3.8242960, -5.8485131, 3.7705231, -9.6567335, 9.6728067
3: -6.6836076, 3.3746753, -6.6568880, 3.3013361, -9.9849434, 10.0315628
4: -6.3953648, 5.0885973, -6.4034963, 5.1084132, -11.5037785, 11.4920940
5: -5.4553452, 4.3688307, -5.4422760, 4.3800688, -9.8354130, 9.8111067
6: -5.1739559, 5.6166935, -5.1251931, 5.6120968, -10.7860527, 10.7418842
7: -6.4849014, 3.8154230, -6.4884343, 3.7428422, -10.2277431, 10.3038578
8: -6.4363947, 4.7033520, -6.3776250, 4.6627674, -11.0991602, 11.0809765
9: -5.1632404, 5.2408319, -5.1638660, 5.2168851, -10.3801250, 10.4046974

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 195

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of NS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_B1_A2_B2_A1_B1

### Relational analysis result of NS_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9315759, upper bound: 10.8938927
time: 3.69 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2

### Relational analysis result of NS_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9232114, upper bound: 10.8852125
time: 2.87 seconds

## BFS NS instance: NS_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.3496671, 4.5350680, -5.9462008, 5.0301385, -10.3798056, 10.4812651
1: -4.2112093, 3.9647894, -4.6917210, 4.3515415, -8.5627508, 8.6565104
2: -5.5114331, 3.6602042, -6.0979252, 3.9255395, -9.4369717, 9.7581291
3: -6.1953759, 3.2098646, -6.9426851, 3.4317157, -9.6270895, 10.1525497
4: -5.9929481, 4.7649536, -6.6729393, 5.3072672, -11.3002148, 11.4378920
5: -5.1525245, 4.1538281, -5.6887264, 4.5628657, -9.7153902, 9.8425541
6: -4.8635740, 5.2468381, -5.3339663, 5.8307419, -10.6943159, 10.5808048
7: -6.0765209, 3.7344501, -6.7502947, 3.9317126, -10.0082340, 10.4847450
8: -6.0151997, 4.4192753, -6.6413298, 4.8465104, -10.8617096, 11.0606031
9: -4.8619461, 4.9255872, -5.3739576, 5.4398603, -10.3018064, 10.2995453

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 62

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of NS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of NS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_B1_A2_B2_A2_A1

### Relational analysis result of NS_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1175657, upper bound: 11.0082028
time: 3.02 seconds

## Relational analysis of NS_B1_A2_B2_A2_A2

### Relational analysis result of NS_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1172932, upper bound: 11.0080611
time: 3.82 seconds

## BFS NS instance: NS_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.4639344, 3.7627633, -4.6506619, 3.9273903, -8.3913250, 8.4134254
1: -3.4279914, 3.3329320, -3.5918355, 3.4686661, -6.8966575, 6.9247675
2: -4.5545049, 3.0240276, -4.7561903, 3.1554313, -7.7099361, 7.7802181
3: -5.1531153, 2.6677372, -5.3846388, 2.7772379, -7.9303532, 8.0523758
4: -4.9622030, 4.0243893, -5.1896172, 4.1842041, -9.1464071, 9.2140064
5: -4.1852784, 3.4526145, -4.3880510, 3.5961809, -7.7814593, 7.8406658
6: -4.0353599, 4.4334331, -4.2093425, 4.6101899, -8.6455498, 8.6427755
7: -5.0743918, 2.9091270, -5.2906771, 3.0606380, -8.1350298, 8.1998043
8: -5.0045600, 3.6973698, -5.2177114, 3.8465910, -8.8511505, 8.9150810
9: -4.0385876, 4.0698185, -4.2104707, 4.2491322, -8.2877197, 8.2802887

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of NS_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_B2_A1_B1_A1_B1

### Relational analysis result of NS_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1261388, upper bound: 11.1259397
time: 6.27 seconds

## Relational analysis of NS_B2_A1_B1_A1_B2

### Relational analysis result of NS_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1259924, upper bound: 11.1257011
time: 4.05 seconds

## BFS NS instance: NS_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.4039345, 4.5828538, -4.6506619, 3.9273903, -9.3313246, 9.2335157
1: -4.2620926, 4.0085993, -3.5918355, 3.4686661, -7.7307577, 7.6004348
2: -5.5749097, 3.6982975, -4.7561903, 3.1554313, -8.7303400, 8.4544878
3: -6.2691259, 3.2524133, -5.3846388, 2.7772379, -9.0463619, 8.6370525
4: -6.0545640, 4.8072472, -5.1896172, 4.1842041, -10.2387676, 9.9968643
5: -5.2149382, 4.1953735, -4.3880510, 3.5961809, -8.8111191, 8.5834236
6: -4.9176188, 5.3008947, -4.2093425, 4.6101899, -9.5278091, 9.5102367
7: -6.1430016, 3.7921386, -5.2906771, 3.0606380, -9.2036400, 9.0828142
8: -6.0835600, 4.4624348, -5.2177114, 3.8465910, -9.9301500, 9.6801462
9: -4.9121599, 4.9810410, -4.2104707, 4.2491322, -9.1612921, 9.1915112

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_B2_A1_B1_A2_A1

### Relational analysis result of NS_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1245937, upper bound: 11.1245630
time: 3.90 seconds

## Relational analysis of NS_B2_A1_B1_A2_A2

### Relational analysis result of NS_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1248072, upper bound: 11.1246574
time: 3.59 seconds

## BFS NS instance: NS_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.4639344, 3.7627633, -5.6321340, 4.7739940, -9.2379284, 9.3948975
1: -3.4279914, 3.3329320, -4.4582376, 4.1670780, -7.5950694, 7.7911692
2: -4.5545049, 3.0240276, -5.8096819, 3.8521299, -8.4066353, 8.8337078
3: -5.1531153, 2.6677372, -6.5404949, 3.3813059, -8.5344210, 9.2082310
4: -4.9622030, 4.0243893, -6.3185225, 4.9957252, -9.9579277, 10.3429117
5: -4.1852784, 3.4526145, -5.4498758, 4.3682470, -8.5535259, 8.9024906
6: -4.0353599, 4.4334331, -5.1249552, 5.5032926, -9.5386524, 9.5583878
7: -5.0743918, 2.9091270, -6.3899922, 3.9725924, -9.0469837, 9.2991190
8: -5.0045600, 3.6973698, -6.3383703, 4.6399903, -9.6445503, 10.0357399
9: -4.0385876, 4.0698185, -5.1139555, 5.1916995, -9.2302876, 9.1837740

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 167

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_B2_A1_B2_A1_B1

### Relational analysis result of NS_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1243888, upper bound: 11.1258433
time: 3.57 seconds

## Relational analysis of NS_B2_A1_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1248072, upper bound: 11.1262509
time: 4.12 seconds

## BFS NS instance: NS_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.4039345, 4.5828538, -5.6321340, 4.7739940, -10.1779270, 10.2149878
1: -4.2620926, 4.0085993, -4.4582376, 4.1670780, -8.4291706, 8.4668360
2: -5.5749097, 3.6982975, -5.8096819, 3.8521299, -9.4270391, 9.5079784
3: -6.2691259, 3.2524133, -6.5404949, 3.3813059, -9.6504316, 9.7929068
4: -6.0545640, 4.8072472, -6.3185225, 4.9957252, -11.0502872, 11.1257696
5: -5.2149382, 4.1953735, -5.4498758, 4.3682470, -9.5831852, 9.6452494
6: -4.9176188, 5.3008947, -5.1249552, 5.5032926, -10.4209108, 10.4258490
7: -6.1430016, 3.7921386, -6.3899922, 3.9725924, -10.1155939, 10.1821289
8: -6.0835600, 4.4624348, -6.3383703, 4.6399903, -10.7235508, 10.8008041
9: -4.9121599, 4.9810410, -5.1139555, 5.1916995, -10.1038589, 10.0949965

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_B2_A1_B2_A2_B1

### Relational analysis result of NS_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1261489, upper bound: 11.1299233
time: 3.51 seconds

## Relational analysis of NS_B2_A1_B2_A2_B2

### Relational analysis result of NS_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1261489, upper bound: 11.1299875
time: 4.30 seconds

## BFS NS instance: NS_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.8584981, 3.2181478, -4.3444147, 3.6630158, -7.5215139, 7.5625625
1: -2.8772225, 2.8652768, -3.3237481, 3.2465346, -6.1237574, 6.1890249
2: -3.8263783, 2.5537817, -4.4240656, 2.9369316, -6.7633100, 6.9778471
3: -4.3763485, 2.2742977, -5.0061455, 2.5933790, -6.9697275, 7.2804432
4: -4.1753302, 3.4981897, -4.8212271, 3.9263148, -8.1016445, 8.3194170
5: -3.5086632, 2.9527082, -4.0586762, 3.3646750, -6.8733382, 7.0113845
6: -3.4602578, 3.8353443, -3.9242003, 4.3213425, -7.7816000, 7.7595444
7: -4.3166809, 2.5465417, -4.9383984, 2.8050406, -7.1217213, 7.4849401
8: -4.2752895, 3.1826899, -4.8682251, 3.6020455, -7.8773351, 8.0509148
9: -3.4437122, 3.4383764, -3.9307845, 3.9556148, -7.3993273, 7.3691607

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_B2_A2_B1_A1_B1

### Relational analysis result of NS_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1240400, upper bound: 11.1241512
time: 3.74 seconds

## Relational analysis of NS_B2_A2_B1_A1_B2

### Relational analysis result of NS_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1238903, upper bound: 11.1238903
time: 3.88 seconds

## BFS NS instance: NS_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.5584850, 3.8303990, -4.3444147, 3.6630158, -8.2215004, 8.1748142
1: -3.5020332, 3.3843598, -3.3237481, 3.2465346, -6.7485676, 6.7081079
2: -4.6381016, 3.0409908, -4.4240656, 2.9369316, -7.5750332, 7.4650564
3: -5.2599850, 2.6985364, -5.0061455, 2.5933790, -7.8533640, 7.7046819
4: -5.0237656, 4.0985851, -4.8212271, 3.9263148, -8.9500809, 8.9198122
5: -4.2589765, 3.4976873, -4.0586762, 3.3646750, -7.6236515, 7.5563636
6: -4.0969558, 4.5319109, -3.9242003, 4.3213425, -8.4182987, 8.4561110
7: -5.1691523, 2.9202616, -4.9383984, 2.8050406, -7.9741926, 7.8586597
8: -5.1048861, 3.7618594, -4.8682251, 3.6020455, -8.7069321, 8.6300850
9: -4.1117144, 4.1464911, -3.9307845, 3.9556148, -8.0673294, 8.0772753

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 188

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of NS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_B2_A2_B1_A2_A1

### Relational analysis result of NS_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1241512, upper bound: 11.1248668
time: 3.67 seconds

## Relational analysis of NS_B2_A2_B1_A2_A2

### Relational analysis result of NS_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1238903, upper bound: 11.1238903
time: 3.10 seconds

## BFS NS instance: NS_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.8584981, 3.2181478, -5.2550688, 4.4531546, -8.3116531, 8.4732151
1: -2.8772225, 2.8652768, -4.1278057, 3.8999507, -6.7771730, 6.9930825
2: -3.8263783, 2.5537817, -5.4177375, 3.5898154, -7.4161940, 7.9715195
3: -4.3763485, 2.2742977, -6.0902853, 3.1582687, -7.5346174, 8.3645821
4: -4.1753302, 3.4981897, -5.8840542, 4.6870537, -8.8623838, 9.3822441
5: -3.5086632, 2.9527082, -5.0547428, 4.0843000, -7.5929632, 8.0074511
6: -3.4602578, 3.8353443, -4.7750130, 5.1669002, -8.6271582, 8.6103573
7: -4.3166809, 2.5465417, -5.9776363, 3.6566114, -7.9732924, 8.5241776
8: -4.2752895, 3.1826899, -5.9084744, 4.3449755, -8.6202650, 9.0911636
9: -3.4437122, 3.4383764, -4.7770643, 4.8386016, -8.2823143, 8.2154398

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_B2_A2_B2_A1_B1

### Relational analysis result of NS_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1241491, upper bound: 11.1253344
time: 2.47 seconds

## Relational analysis of NS_B2_A2_B2_A1_B2

### Relational analysis result of NS_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1241622, upper bound: 11.1253464
time: 2.77 seconds

## BFS NS instance: NS_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.5584850, 3.8303990, -5.2550688, 4.4531546, -9.0116396, 9.0854673
1: -3.5020332, 3.3843598, -4.1278057, 3.8999507, -7.4019842, 7.5121655
2: -4.6381016, 3.0409908, -5.4177375, 3.5898154, -8.2279167, 8.4587278
3: -5.2599850, 2.6985364, -6.0902853, 3.1582687, -8.4182539, 8.7888203
4: -5.0237656, 4.0985851, -5.8840542, 4.6870537, -9.7108173, 9.9826393
5: -4.2589765, 3.4976873, -5.0547428, 4.0843000, -8.3432770, 8.5524302
6: -4.0969558, 4.5319109, -4.7750130, 5.1669002, -9.2638559, 9.3069229
7: -5.1691523, 2.9202616, -5.9776363, 3.6566114, -8.8257627, 8.8978977
8: -5.1048861, 3.7618594, -5.9084744, 4.3449755, -9.4498615, 9.6703339
9: -4.1117144, 4.1464911, -4.7770643, 4.8386016, -8.9503155, 8.9235554

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 62

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_B2_A2_B2_A2_B1

### Relational analysis result of NS_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1220729, upper bound: 11.1266882
time: 3.68 seconds

## Relational analysis of NS_B2_A2_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1224174, upper bound: 11.1269788
time: 3.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 8.14 seconds
NS_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 8.14
Output dim: 7, lower bound: -10.9076481, upper bound: 10.8803418
NS_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 8.14
Output dim: 7, lower bound: -10.8994532, upper bound: 10.8733505
NS_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1168698, upper bound: 11.0060424
NS_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1164730, upper bound: 11.0057913
NS_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1164038, upper bound: 11.0073628
NS_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1164038, upper bound: 11.0073628
NS_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1164038, upper bound: 11.0073627
NS_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1164038, upper bound: 11.0073627
NS_B1_A2_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.0135655, upper bound: 10.9362424
NS_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1193433, upper bound: 11.0869943
NS_B1_A2_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 8.14
Output dim: 7, lower bound: -10.9597565, upper bound: 10.9017391
NS_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1193779, upper bound: 11.0869746
NS_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 8.14
Output dim: 7, lower bound: -10.9315759, upper bound: 10.8938927
NS_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 8.14
Output dim: 7, lower bound: -10.9232114, upper bound: 10.8852125
NS_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1175657, upper bound: 11.0082028
NS_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1172932, upper bound: 11.0080611
NS_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1261388, upper bound: 11.1259397
NS_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1259924, upper bound: 11.1257011
NS_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1245937, upper bound: 11.1245630
NS_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1248072, upper bound: 11.1246574
NS_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1243888, upper bound: 11.1258433
NS_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1248072, upper bound: 11.1262509
NS_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1261489, upper bound: 11.1299233
NS_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1261489, upper bound: 11.1299875
NS_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1240400, upper bound: 11.1241512
NS_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1238903, upper bound: 11.1238903
NS_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1241512, upper bound: 11.1248668
NS_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1238903, upper bound: 11.1238903
NS_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1241491, upper bound: 11.1253344
NS_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1241622, upper bound: 11.1253464
NS_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1220729, upper bound: 11.1266882
NS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 8.14
Output dim: 7, lower bound: -11.1224174, upper bound: 11.1269788

## BFS NS instance: NS_B1_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -3.9492617, 3.3225079, -5.1096673, 4.2909670, -8.2402287, 8.4321747
1: -2.9739110, 2.9505847, -3.9767737, 3.7683315, -6.7422428, 6.9273586
2: -3.9727244, 2.6676846, -5.1801553, 3.3482504, -7.3209748, 7.8478398
3: -4.5059576, 2.3508911, -5.9644470, 2.9513569, -7.4573145, 8.3153381
4: -4.3417745, 3.5873179, -5.6998339, 4.6034484, -8.9452229, 9.2871513
5: -3.6429276, 3.0598085, -4.7842011, 3.8893666, -7.5322943, 7.8440094
6: -3.5634487, 3.9306087, -4.5871773, 5.0449390, -8.6083879, 8.5177860
7: -4.4625702, 2.5388021, -5.7944584, 3.1894038, -7.6519737, 8.3332605
8: -4.4096813, 3.2768176, -5.6959181, 4.1682644, -8.5779457, 8.9727354
9: -3.5624352, 3.5667450, -4.5976634, 4.6180534, -8.1804886, 8.1644087

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_B1_A1_B1_A2_A1_B1

### Relational analysis result of NS_B1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1164730, upper bound: 11.0057913
time: 4.82 seconds

## Relational analysis of NS_B1_A1_B1_A2_A1_B2

### Relational analysis result of NS_B1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1164730, upper bound: 11.0057913
time: 4.02 seconds

## BFS NS instance: NS_B1_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -4.3264685, 3.6196613, -5.0355129, 4.2248430, -8.5513115, 8.6551743
1: -3.2899902, 3.2147391, -3.9133272, 3.7148876, -7.0048780, 7.1280661
2: -4.3557467, 2.8670359, -5.0983210, 3.2996349, -7.6553817, 7.9653568
3: -4.9862804, 2.5450547, -5.8723111, 2.9100783, -7.8963585, 8.4173660
4: -4.7601867, 3.9001796, -5.6099405, 4.5395184, -9.2997055, 9.5101204
5: -3.9925668, 3.2992568, -4.7032948, 3.8304002, -7.8229671, 8.0025520
6: -3.8892434, 4.2864451, -4.5207281, 4.9744492, -8.8636923, 8.8071728
7: -4.8717508, 2.6595528, -5.7081633, 3.1304951, -8.0022459, 8.3677158
8: -4.8308635, 3.5596609, -5.6119642, 4.1104703, -8.9413338, 9.1716251
9: -3.8794930, 3.8948932, -4.5288715, 4.5465961, -8.4260893, 8.4237652

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 167

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_B1_A1_B1_A2_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1164730, upper bound: 11.0057913
time: 4.00 seconds

## Relational analysis of NS_B1_A1_B1_A2_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1164730, upper bound: 11.0057913
time: 4.62 seconds

## BFS NS instance: NS_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -4.0873866, 3.4428470, -5.4494925, 4.6088858, -8.6962719, 8.8923397
1: -3.0969081, 3.0533171, -4.2731528, 4.0089622, -7.1058702, 7.3264699
2: -4.1307240, 2.7639723, -5.5659952, 3.5977566, -7.7284803, 8.3299675
3: -4.6819892, 2.4352441, -6.3565207, 3.1551204, -7.8371096, 8.7917652
4: -4.5121260, 3.7064044, -6.1090908, 4.8896503, -9.4017763, 9.8154955
5: -3.7926674, 3.1703086, -5.1782761, 4.1790099, -7.9716773, 8.3485851
6: -3.6902213, 4.0671558, -4.8949704, 5.3730335, -9.0632553, 8.9621258
7: -4.6297941, 2.5990574, -6.1977100, 3.5352116, -8.1650057, 8.7967672
8: -4.5704861, 3.3888319, -6.0882902, 4.4527435, -9.0232296, 9.4771223
9: -3.6935313, 3.7052958, -4.9323845, 4.9717121, -8.6652431, 8.6376801

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 107

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_B1_A1_B2_B1_A1_B1

### Relational analysis result of NS_B1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1152128, upper bound: 11.0965537
time: 3.43 seconds

## Relational analysis of NS_B1_A1_B2_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1151623, upper bound: 11.0887778
time: 3.64 seconds

## BFS NS instance: NS_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -4.4660363, 3.7413750, -5.4494925, 4.6088858, -9.0749226, 9.1908674
1: -3.4149590, 3.3184099, -4.2731528, 4.0089622, -7.4239211, 7.5915627
2: -4.5153646, 2.9646833, -5.5659952, 3.5977566, -8.1131210, 8.5306787
3: -5.1651483, 2.6303575, -6.3565207, 3.1551204, -8.3202686, 8.9868784
4: -4.9330158, 4.0203042, -6.1090908, 4.8896503, -9.8226662, 10.1293945
5: -4.1436954, 3.4106140, -5.1782761, 4.1790099, -8.3227053, 8.5888901
6: -4.0174942, 4.4248686, -4.8949704, 5.3730335, -9.3905277, 9.3198395
7: -5.0408797, 2.7882328, -6.1977100, 3.5352116, -8.5760918, 8.9859428
8: -4.9943237, 3.6726134, -6.0882902, 4.4527435, -9.4470673, 9.7609034
9: -4.0120726, 4.0356197, -4.9323845, 4.9717121, -8.9837847, 8.9680042

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_B1_A1_B2_B1_A2_B1

### Relational analysis result of NS_B1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1181690, upper bound: 11.1172819
time: 4.48 seconds

## Relational analysis of NS_B1_A1_B2_B1_A2_B2

### Relational analysis result of NS_B1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1180453, upper bound: 11.1170204
time: 4.08 seconds

## BFS NS instance: NS_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -4.0873866, 3.4428470, -5.8768077, 4.9431505, -9.0305367, 9.3196545
1: -3.0969081, 3.0533171, -4.6315069, 4.3059874, -7.4028955, 7.6848240
2: -4.1307240, 2.7639723, -5.9966087, 3.8255706, -7.9562945, 8.7605810
3: -4.6819892, 2.4352441, -6.8961678, 3.3764992, -8.0584888, 9.3314114
4: -4.5121260, 3.7064044, -6.5799365, 5.2423944, -9.7545204, 10.2863407
5: -3.7926674, 3.1703086, -5.5736613, 4.4555941, -8.2482615, 8.7439699
6: -3.6902213, 4.0671558, -5.2616062, 5.7724109, -9.4626322, 9.3287621
7: -4.6297941, 2.5990574, -6.6583509, 3.7552495, -8.3850441, 9.2574081
8: -4.5704861, 3.3888319, -6.5595951, 4.7728205, -9.3433065, 9.9484272
9: -3.6935313, 3.7052958, -5.2905622, 5.3465266, -9.0400581, 8.9958582

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_B1_A1_B2_B2_A1_B1

### Relational analysis result of NS_B1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1146357, upper bound: 10.9954023
time: 3.38 seconds

## Relational analysis of NS_B1_A1_B2_B2_A1_B2

### Relational analysis result of NS_B1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1142501, upper bound: 10.9808119
time: 3.88 seconds

## BFS NS instance: NS_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -4.4660363, 3.7413750, -5.8768077, 4.9431505, -9.4091873, 9.6181831
1: -3.4149590, 3.3184099, -4.6315069, 4.3059874, -7.7209463, 7.9499168
2: -4.5153646, 2.9646833, -5.9966087, 3.8255706, -8.3409348, 8.9612923
3: -5.1651483, 2.6303575, -6.8961678, 3.3764992, -8.5416470, 9.5265255
4: -4.9330158, 4.0203042, -6.5799365, 5.2423944, -10.1754103, 10.6002407
5: -4.1436954, 3.4106140, -5.5736613, 4.4555941, -8.5992889, 8.9842758
6: -4.0174942, 4.4248686, -5.2616062, 5.7724109, -9.7899055, 9.6864748
7: -5.0408797, 2.7882328, -6.6583509, 3.7552495, -8.7961292, 9.4465837
8: -4.9943237, 3.6726134, -6.5595951, 4.7728205, -9.7671442, 10.2322083
9: -4.0120726, 4.0356197, -5.2905622, 5.3465266, -9.3585987, 9.3261814

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_B1_A1_B2_B2_A2_B1

### Relational analysis result of NS_B1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1146357, upper bound: 10.9954023
time: 2.49 seconds

## Relational analysis of NS_B1_A1_B2_B2_A2_B2

### Relational analysis result of NS_B1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1142501, upper bound: 10.9808119
time: 3.55 seconds

## BFS NS instance: NS_B1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -4.1006732, 3.4765601, -4.9442382, 4.1477203, -8.2483921, 8.4207983
1: -3.1271696, 3.0710406, -3.8339763, 3.6507661, -6.7779360, 6.9050169
2: -4.1642666, 2.8017714, -5.0002880, 3.2426953, -7.4069614, 7.8020592
3: -4.6820593, 2.4623525, -5.7538104, 2.8608918, -7.5429511, 8.2161627
4: -4.5397091, 3.7169251, -5.5037842, 4.4599276, -8.9996367, 9.2207088
5: -3.8016036, 3.2045059, -4.6010432, 3.7626524, -7.5642557, 7.8055487
6: -3.7080681, 4.0847898, -4.4400868, 4.8867931, -8.5948610, 8.5248766
7: -4.6635189, 2.6373219, -5.6022453, 3.0568135, -7.7203321, 8.2395668
8: -4.5919671, 3.4288514, -5.5068407, 4.0415030, -8.6334686, 8.9356918
9: -3.7229123, 3.7322321, -4.4435339, 4.4605536, -8.1834660, 8.1757660

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_B1_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_B1_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_B1_A2_B1_A1_A2_A1

### Relational analysis result of NS_B1_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1177881, upper bound: 11.0641470
time: 4.23 seconds

## Relational analysis of NS_B1_A2_B1_A1_A2_A2

### Relational analysis result of NS_B1_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1176609, upper bound: 11.0640514
time: 3.23 seconds

## BFS NS instance: NS_B1_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -4.6328616, 3.9213600, -4.8386889, 4.0582848, -8.6911469, 8.7600489
1: -3.5866861, 3.4487538, -3.7442331, 3.5751555, -7.1618414, 7.1929870
2: -4.7305923, 3.1309655, -4.8879538, 3.1801243, -7.9107161, 8.0189190
3: -5.3555412, 2.7515800, -5.6208000, 2.8043032, -8.1598444, 8.3723803
4: -5.1652727, 4.1766024, -5.3806238, 4.3700671, -9.5353394, 9.5572252
5: -4.3495979, 3.5886021, -4.4956808, 3.6870666, -8.0366650, 8.0842829
6: -4.1831307, 4.5936866, -4.3474541, 4.7861338, -8.9692621, 8.9411411
7: -5.2720003, 3.0119145, -5.4824762, 2.9912453, -8.2632456, 8.4943905
8: -5.1951847, 3.8370492, -5.3890414, 3.9604700, -9.1556549, 9.2260904
9: -4.2001848, 4.2264462, -4.3509970, 4.3637633, -8.5639477, 8.5774431

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_B1_A2_B1_A2_A2_A1

### Relational analysis result of NS_B1_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1177441, upper bound: 11.0640937
time: 4.15 seconds

## Relational analysis of NS_B1_A2_B1_A2_A2_A2

### Relational analysis result of NS_B1_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1176203, upper bound: 11.0640160
time: 3.60 seconds

## BFS NS instance: NS_B1_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -4.7970834, 4.0584536, -5.8434377, 4.9448452, -9.7419281, 9.9018898
1: -3.7191308, 3.5656333, -4.6044922, 4.2814097, -8.0005398, 8.1701260
2: -4.9230547, 3.2609437, -5.9910841, 3.8551493, -8.7782021, 9.2520256
3: -5.5281062, 2.8743742, -6.8227839, 3.3732595, -8.9013653, 9.6971550
4: -5.3432655, 4.2987080, -6.5575895, 5.2216682, -10.5649338, 10.8562965
5: -4.5641670, 3.7305734, -5.5837936, 4.4836617, -9.0478287, 9.3143663
6: -4.3458581, 4.7371998, -5.2432590, 5.7380590, -10.0839167, 9.9804592
7: -5.4490590, 3.2621841, -6.6382532, 3.8485498, -9.2976065, 9.9004364
8: -5.3788939, 3.9785070, -6.5278563, 4.7661371, -10.1450310, 10.5063629
9: -4.3602448, 4.3962474, -5.2843909, 5.3441119, -9.7043571, 9.6806374

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 195

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_B1_A2_B2_A2_A1_B1

### Relational analysis result of NS_B1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1172932, upper bound: 11.0080611
time: 3.11 seconds

## Relational analysis of NS_B1_A2_B2_A2_A1_B2

### Relational analysis result of NS_B1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1172932, upper bound: 11.0080611
time: 3.90 seconds

## BFS NS instance: NS_B1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -5.2195163, 4.3918262, -5.7471142, 4.8630018, -10.0825176, 10.1389399
1: -4.0765710, 3.8597212, -4.5231724, 4.2161903, -8.2927608, 8.3828936
2: -5.3540530, 3.4833031, -5.8887835, 3.7896214, -9.1436749, 9.3720865
3: -6.0681243, 3.0910501, -6.7104845, 3.3203616, -9.3884850, 9.8015337
4: -5.8112364, 4.6537199, -6.4493570, 5.1408339, -10.9520702, 11.1030769
5: -4.9561791, 4.0012355, -5.4837065, 4.4079123, -9.3640919, 9.4849415
6: -4.7125511, 5.1351395, -5.1581373, 5.6509690, -10.3635197, 10.2932768
7: -5.9090552, 3.4814317, -6.5333486, 3.7668359, -9.6758909, 10.0147781
8: -5.8506641, 4.2947741, -6.4210882, 4.6898856, -10.5405502, 10.7158613
9: -4.7193813, 4.7707734, -5.1990566, 5.2531371, -9.9725170, 9.9698286

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_B1_A2_B2_A2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1172932, upper bound: 11.0080611
time: 2.88 seconds

## Relational analysis of NS_B1_A2_B2_A2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1172932, upper bound: 11.0080611
time: 3.39 seconds

## BFS NS instance: NS_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.3682775, 3.6805770, -4.1551285, 3.4987626, -7.8670402, 7.8357058
1: -3.3437903, 3.2633688, -3.1560695, 3.1040893, -6.4478798, 6.4194384
2: -4.4501853, 2.9550147, -4.2075605, 2.8088052, -7.2589903, 7.1625752
3: -5.0356240, 2.6101062, -4.7686262, 2.4788713, -7.5144954, 7.3787327
4: -4.8472338, 3.9431956, -4.5917883, 3.7620959, -8.6093292, 8.5349836
5: -4.0844984, 3.3799300, -3.8629913, 3.2195601, -7.3040586, 7.2429214
6: -3.9470522, 4.3422165, -3.7525716, 4.1328053, -8.0798578, 8.0947876
7: -4.9627118, 2.8301222, -4.7091312, 2.6554952, -7.6182070, 7.5392532
8: -4.8949733, 3.6205847, -4.6487055, 3.4439297, -8.3389034, 8.2692900
9: -3.9511564, 3.9781313, -3.7547159, 3.7711174, -7.7222738, 7.7328472

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1259924, upper bound: 11.1257011
time: 3.61 seconds

## Relational analysis of NS_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1259924, upper bound: 11.1257011
time: 4.24 seconds

## BFS NS instance: NS_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.2753887, 3.6005721, -4.5313091, 3.7958052, -8.0711937, 8.1318817
1: -3.2615232, 3.1950850, -3.4723837, 3.3673162, -6.6288395, 6.6674690
2: -4.3465919, 2.8888903, -4.5899277, 3.0086913, -7.3552833, 7.4788179
3: -4.9201536, 2.5545230, -5.2495584, 2.6726487, -7.5928020, 7.8040814
4: -4.7335691, 3.8645372, -5.0098124, 4.0743980, -8.8079672, 8.8743496
5: -3.9868550, 3.3070359, -4.2118816, 3.4585290, -7.4453840, 7.5189176
6: -3.8620567, 4.2528319, -4.0778475, 4.4885473, -8.3506041, 8.3306789
7: -4.8536730, 2.7501113, -5.1181426, 2.8440058, -7.6976786, 7.8682537
8: -4.7884097, 3.5448413, -5.0703616, 3.7258093, -8.5142193, 8.6152029
9: -3.8644795, 3.8886828, -4.0713291, 4.1003075, -7.9647870, 7.9600120

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 62

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1259924, upper bound: 11.1257011
time: 3.38 seconds

## Relational analysis of NS_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1259924, upper bound: 11.1257011
time: 3.26 seconds

## BFS NS instance: NS_B2_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -4.1556058, 3.5204816, -4.3517971, 3.6712892, -7.8268938, 7.8722787
1: -3.1775775, 3.1138961, -3.3343387, 3.2533774, -6.4309535, 6.4482346
2: -4.2328587, 2.8387518, -4.4297342, 2.9469829, -7.1798415, 7.2684860
3: -4.7552881, 2.5043960, -5.0176296, 2.5989733, -7.3542614, 7.5220256
4: -4.6037407, 3.7624521, -4.8351526, 3.9318855, -8.5356264, 8.5976048
5: -3.8651345, 3.2464995, -4.0615368, 3.3707361, -7.2358704, 7.3080354
6: -3.7594581, 4.1414080, -3.9322724, 4.3267322, -8.0861902, 8.0736809
7: -4.7309337, 2.6980491, -4.9459796, 2.7996655, -7.5305991, 7.6440287
8: -4.6582708, 3.4734387, -4.8772583, 3.6097689, -8.2680397, 8.3506966
9: -3.7728639, 3.7891417, -3.9350214, 3.9601490, -7.7330127, 7.7241631

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 195

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_B2_A1_B1_A2_A1_B1

### Relational analysis result of NS_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1249119, upper bound: 11.1242453
time: 4.39 seconds

## Relational analysis of NS_B2_A1_B1_A2_A1_B2

### Relational analysis result of NS_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1249119, upper bound: 11.1245612
time: 4.42 seconds

## BFS NS instance: NS_B2_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -4.6915998, 3.9681177, -4.2279429, 3.5677762, -8.2593765, 8.1960602
1: -3.6395550, 3.4945710, -3.2271152, 3.1637592, -6.8033133, 6.7216864
2: -4.8017235, 3.1683235, -4.2973509, 2.8650041, -7.6667275, 7.4656744
3: -5.4341679, 2.7943971, -4.8643198, 2.5285311, -7.9626985, 7.6587172
4: -5.2324781, 4.2259240, -4.6892986, 3.8281102, -9.0605888, 8.9152222
5: -4.4142709, 3.6316025, -3.9386539, 3.2803245, -7.6945953, 7.5702562
6: -4.2385778, 4.6533618, -3.8221295, 4.2083712, -8.4469490, 8.4754915
7: -5.3429294, 3.0678113, -4.8050675, 2.7119708, -8.0549002, 7.8728790
8: -5.2655411, 3.8845620, -4.7368240, 3.5117855, -8.7773256, 8.6213856
9: -4.2526493, 4.2849607, -3.8252387, 3.8446894, -8.0973387, 8.1101990

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 195

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_B2_A1_B1_A2_A2_A1

### Relational analysis result of NS_B2_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1244319, upper bound: 11.1236334
time: 5.61 seconds

## Relational analysis of NS_B2_A1_B1_A2_A2_A2

### Relational analysis result of NS_B2_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1242017, upper bound: 11.1234830
time: 3.83 seconds

## BFS NS instance: NS_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.1753397, 3.5199065, -4.3394361, 3.6792397, -7.8545794, 7.8593421
1: -3.1773787, 3.1250668, -3.3399935, 3.2450721, -6.4224510, 6.4650602
2: -4.2384720, 2.8245883, -4.4294639, 2.9651966, -7.2036686, 7.2540517
3: -4.7958527, 2.4979548, -4.9797826, 2.6093762, -7.4052286, 7.4777369
4: -4.6170616, 3.7819247, -4.8261728, 3.9197516, -8.5368118, 8.6080971
5: -3.8789358, 3.2325814, -4.0602946, 3.3897307, -7.2686663, 7.2928762
6: -3.7725761, 4.1581774, -3.9258242, 4.3135891, -8.0861654, 8.0840015
7: -4.7413683, 2.6581452, -4.9405947, 2.8486462, -7.5900140, 7.5987396
8: -4.6741538, 3.4695315, -4.8664880, 3.6171536, -8.2913074, 8.3360195
9: -3.7716186, 3.7929673, -3.9418044, 3.9632845, -7.7349033, 7.7347717

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 195

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_B2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_B2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1257503, upper bound: 11.1258413
time: 8.95 seconds

## Relational analysis of NS_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1257503, upper bound: 11.1258433
time: 5.21 seconds

## BFS NS instance: NS_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.0561905, 3.4192095, -4.8795199, 4.1313295, -8.1875200, 8.2987289
1: -3.0744095, 3.0379171, -3.8055847, 3.6294904, -6.7038999, 6.8435020
2: -4.1078110, 2.7491920, -5.0027056, 3.2993524, -7.4071636, 7.7518964
3: -4.6461248, 2.4302998, -5.6658282, 2.9024756, -7.5486002, 8.0961285
4: -4.4761209, 3.6812022, -5.4608669, 4.3873472, -8.8634682, 9.1420689
5: -3.7603040, 3.1459899, -4.6150742, 3.7791255, -7.5394297, 7.7610641
6: -3.6665049, 4.0423942, -4.4111543, 4.8288364, -8.4953413, 8.4535484
7: -4.6040525, 2.5962906, -5.5575113, 3.2235613, -7.8276134, 8.1538019
8: -4.5384035, 3.3744876, -5.4796634, 4.0321054, -8.5705090, 8.8541508
9: -3.6651192, 3.6807888, -4.4259167, 4.4638462, -8.1289654, 8.1067057

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_B2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_B2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_B2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_B2_A1_B2_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1251311, upper bound: 11.1253210
time: 3.19 seconds

## Relational analysis of NS_B2_A1_B2_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1250660, upper bound: 11.1252419
time: 4.69 seconds

## BFS NS instance: NS_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.2171221, 4.4216337, -4.9556084, 4.1907248, -9.4078465, 9.3772421
1: -4.0953722, 3.8736196, -3.8601236, 3.6803274, -7.7756996, 7.7337432
2: -5.3770857, 3.5656877, -5.0912547, 3.3708429, -8.7479286, 8.6569405
3: -6.0386839, 3.1419854, -5.7129121, 2.9820671, -9.0207510, 8.8548965
4: -5.8319969, 4.6471257, -5.5104322, 4.4198322, -10.2518291, 10.1575584
5: -5.0144334, 4.0526438, -4.7237659, 3.8489625, -8.8633938, 8.7764091
6: -4.7393436, 5.1299176, -4.4815092, 4.8860960, -9.6254396, 9.6114244
7: -5.9324169, 3.6318429, -5.6295824, 3.3880885, -9.3205051, 9.2614250
8: -5.8693161, 4.3153753, -5.5641255, 4.1086535, -9.9779701, 9.8795013
9: -4.7412605, 4.8017292, -4.4951596, 4.5461903, -9.2874508, 9.2968884

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 148

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1299054, upper bound: 11.1299233
time: 15.13 seconds

## Relational analysis of NS_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1299054, upper bound: 11.1299233
time: 3.64 seconds

## BFS NS instance: NS_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.2119408, 4.4151978, -5.4321222, 4.5761843, -9.7881241, 9.8473206
1: -4.0897875, 3.8687677, -4.2671709, 4.0089874, -8.0987749, 8.1359367
2: -5.3698945, 3.5575607, -5.5879750, 3.6428235, -9.0127172, 9.1455326
3: -6.0320320, 3.1374769, -6.3044615, 3.2340572, -9.2660894, 9.4419365
4: -5.8222265, 4.6414032, -6.0402613, 4.8167143, -10.6389399, 10.6816626
5: -5.0056901, 4.0467777, -5.1759949, 4.1739788, -9.1796684, 9.2227726
6: -4.7316928, 5.1251392, -4.8991728, 5.3330474, -10.0647392, 10.0243120
7: -5.9249067, 3.6197541, -6.1534123, 3.6802642, -9.6051693, 9.7731647
8: -5.8620520, 4.3109665, -6.1059074, 4.4744272, -10.3364782, 10.4168730
9: -4.7349396, 4.7945061, -4.9053221, 4.9750614, -9.7100010, 9.6998281

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1299628, upper bound: 11.1299875
time: 3.79 seconds

## Relational analysis of NS_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1299628, upper bound: 11.1299875
time: 4.19 seconds

## BFS NS instance: NS_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.7908018, 3.1621788, -3.8840101, 3.2646868, -7.0554886, 7.0461888
1: -2.8198123, 2.8166130, -2.9178476, 2.9045858, -5.7243981, 5.7344608
2: -3.7524881, 2.5123734, -3.9031873, 2.6256270, -6.3781152, 6.4155607
3: -4.2909870, 2.2381370, -4.4237337, 2.3197644, -6.6107512, 6.6618710
4: -4.0955424, 3.4388349, -4.2620931, 3.5323310, -7.6278734, 7.7009277
5: -3.4415791, 2.9055755, -3.5788019, 3.0111284, -6.4527073, 6.4843774
6: -3.3998401, 3.7701418, -3.5057926, 3.8699474, -7.2697878, 7.2759342
7: -4.2372003, 2.5366428, -4.3901987, 2.5385275, -6.7757277, 6.9268417
8: -4.1984925, 3.1301100, -4.3354101, 3.2247963, -7.4232888, 7.4655199
9: -3.3829432, 3.3756592, -3.5022833, 3.5077078, -6.8906507, 6.8779426

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1238903, upper bound: 11.1238903
time: 3.10 seconds

## Relational analysis of NS_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1238903, upper bound: 11.1238903
time: 30.57 seconds

## BFS NS instance: NS_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.7300072, 3.1114991, -4.2629585, 3.5661435, -7.2961507, 7.3744574
1: -2.7681038, 2.7729392, -3.2362924, 3.1711547, -5.9392586, 6.0092316
2: -3.6852643, 2.4739814, -4.2896619, 2.8272300, -6.5124941, 6.7636433
3: -4.2146130, 2.2053521, -4.9077110, 2.5154457, -6.7300587, 7.1130629
4: -4.0229464, 3.3848131, -4.6842895, 3.8477404, -7.8706865, 8.0691023
5: -3.3802426, 2.8624997, -3.9327900, 3.2543671, -6.6346097, 6.7952900
6: -3.3449416, 3.7114034, -3.8323205, 4.2294121, -7.5743537, 7.5437241
7: -4.1644363, 2.5266285, -4.8041458, 2.6184893, -6.7829256, 7.3307743
8: -4.1293650, 3.0824275, -4.7600098, 3.5099764, -7.6393414, 7.8424373
9: -3.3274546, 3.3183930, -3.8222408, 3.8388610, -7.1663156, 7.1406336

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 167

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1238903, upper bound: 11.1238903
time: 2.25 seconds

## Relational analysis of NS_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1238903, upper bound: 11.1238903
time: 2.74 seconds

## BFS NS instance: NS_B2_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -4.1071835, 3.4335656, -4.2546253, 3.5863299, -7.6935134, 7.6881909
1: -3.1028953, 3.0478246, -3.2445545, 3.1802120, -6.2831073, 6.2923794
2: -4.1216817, 2.7303617, -4.3237619, 2.8756878, -6.9973698, 7.0541239
3: -4.6858463, 2.4301534, -4.8933144, 2.5398736, -7.2257199, 7.3234677
4: -4.4728432, 3.7107863, -4.7122841, 3.8501420, -8.3229847, 8.4230709
5: -3.7868564, 3.1489787, -3.9657276, 3.2961307, -7.0829868, 7.1147060
6: -3.6855490, 4.0841188, -3.8424153, 4.2342415, -7.9197903, 7.9265342
7: -4.6222105, 2.6055369, -4.8328142, 2.7323499, -7.3545604, 7.4383512
8: -4.5776424, 3.3901985, -4.7647896, 3.5289240, -8.1065664, 8.1549883
9: -3.6885529, 3.6994214, -3.8477809, 3.8694921, -7.5580449, 7.5472021

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_B2_A2_B1_A2_A1_B1

### Relational analysis result of NS_B2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1252744, upper bound: 11.1247740
time: 3.79 seconds

## Relational analysis of NS_B2_A2_B1_A2_A1_B2

### Relational analysis result of NS_B2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1252744, upper bound: 11.1247740
time: 3.23 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 6.27 + 597.34 = 603.61 seconds
