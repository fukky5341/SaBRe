## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 189.2309129667


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795)
1: (-85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841)
2: (-113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534)
3: (-120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785)
4: (-110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165)
5: (-99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473)
6: (-95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926)
7: (-103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229)
8: (-124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940)
9: (-94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.25 + 10.57 = 12.82 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -189.4203333, upper bound: 189.4203333

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 249

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4145679, upper bound: 189.4145078
time: 9.62 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4140152, upper bound: 189.4140152
time: 6.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 16.19 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 16.19
Output dim: 7, lower bound: -189.4145679, upper bound: 189.4145078
NS_A2, status: Status.UNKNOWN, split count: 1, time: 16.19
Output dim: 7, lower bound: -189.4140152, upper bound: 189.4140152

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -93.7176285, 74.2614059, -103.0916901, 81.6635971, -175.3812103, 177.3530884
1: -77.9373627, 66.0082703, -85.8328323, 72.6308670, -150.5682220, 151.8410950
2: -102.6867447, 67.7566452, -113.0277863, 74.4069977, -177.0937042, 180.7844238
3: -109.3327103, 58.4988251, -120.3313828, 64.2784958, -173.6112061, 178.8302002
4: -99.9435730, 77.2694244, -110.0417786, 84.9963455, -184.9399109, 187.3112030
5: -90.1656723, 70.4592361, -99.1850357, 77.4622269, -167.6278839, 169.6442719
6: -86.4557037, 82.5762329, -95.0854187, 90.8557663, -177.3114624, 177.6616516
7: -93.9075470, 79.5684967, -103.3243332, 87.4172974, -181.3248444, 182.8928223
8: -112.8780136, 77.0512695, -124.1503143, 84.7329102, -197.6109314, 201.2015839
9: -85.6821136, 84.4675827, -94.2049332, 92.8833542, -178.5654602, 178.6725159

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 249

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4140125, upper bound: 189.4140125
time: 8.47 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4140125, upper bound: 189.4140127
time: 7.26 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -92.2015152, 73.0499954, -98.4734726, 78.0131302, -170.2146454, 171.5234375
1: -76.6287842, 64.9156952, -81.9517822, 69.3743591, -146.0031433, 146.8674469
2: -101.0087738, 66.6834793, -107.9397125, 71.1324615, -172.1412354, 174.6231842
3: -107.5456848, 57.5384598, -114.9146881, 61.4352531, -168.9808960, 172.4531403
4: -98.2835770, 75.9801331, -105.0641785, 81.1904602, -179.4740295, 181.0443115
5: -88.6693954, 69.2845078, -94.7389145, 74.0083542, -162.6777496, 164.0234222
6: -85.0473480, 81.2346573, -90.8335648, 86.7774963, -171.8248444, 172.0682068
7: -92.3694611, 78.2922058, -98.6884918, 83.5497208, -175.9191895, 176.9806976
8: -111.0548859, 75.7827454, -118.5953751, 80.9447098, -191.9996033, 194.3781128
9: -84.2934799, 83.0959854, -90.0054398, 88.7362442, -173.0296936, 173.1014099

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4114571, upper bound: 189.4114110
time: 6.66 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4114398, upper bound: 189.4114398
time: 6.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 15.55 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 15.55
Output dim: 7, lower bound: -189.4140125, upper bound: 189.4140125
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 15.55
Output dim: 7, lower bound: -189.4140125, upper bound: 189.4140127
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 15.55
Output dim: 7, lower bound: -189.4114571, upper bound: 189.4114110
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 15.55
Output dim: 7, lower bound: -189.4114398, upper bound: 189.4114398

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -93.7176285, 74.2614059, -93.7176285, 74.2614059, -167.9790344, 167.9790344
1: -77.9373627, 66.0082703, -77.9373627, 66.0082703, -143.9456329, 143.9456329
2: -102.6867447, 67.7566452, -102.6867447, 67.7566452, -170.4433746, 170.4433746
3: -109.3327103, 58.4988251, -109.3327103, 58.4988251, -167.8315430, 167.8315430
4: -99.9435730, 77.2694244, -99.9435730, 77.2694244, -177.2129974, 177.2129974
5: -90.1656723, 70.4592361, -90.1656723, 70.4592361, -160.6249084, 160.6249084
6: -86.4557037, 82.5762329, -86.4557037, 82.5762329, -169.0319366, 169.0319366
7: -93.9075470, 79.5684967, -93.9075470, 79.5684967, -173.4760437, 173.4760437
8: -112.8780136, 77.0512695, -112.8780136, 77.0512695, -189.9292908, 189.9292908
9: -85.6821136, 84.4675827, -85.6821136, 84.4675827, -170.1496887, 170.1496887

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4118914, upper bound: 189.4119851
time: 8.45 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4120178, upper bound: 189.4120472
time: 8.19 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -93.7176285, 74.2614059, -92.2015152, 73.0499954, -166.7675934, 166.4629059
1: -77.9373627, 66.0082703, -76.6287842, 64.9156952, -142.8530273, 142.6370544
2: -102.6867447, 67.7566452, -101.0087738, 66.6834793, -169.3701935, 168.7654114
3: -109.3327103, 58.4988251, -107.5456848, 57.5384598, -166.8711700, 166.0444641
4: -99.9435730, 77.2694244, -98.2835770, 75.9801331, -175.9237061, 175.5529938
5: -90.1656723, 70.4592361, -88.6693954, 69.2845078, -159.4501648, 159.1286316
6: -86.4557037, 82.5762329, -85.0473480, 81.2346573, -167.6903687, 167.6235809
7: -93.9075470, 79.5684967, -92.3694611, 78.2922058, -172.1997528, 171.9379578
8: -112.8780136, 77.0512695, -111.0548859, 75.7827454, -188.6607666, 188.1061554
9: -85.6821136, 84.4675827, -84.2934799, 83.0959854, -168.7781067, 168.7610626

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4118914, upper bound: 189.4119852
time: 7.62 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4120178, upper bound: 189.4120472
time: 8.07 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -89.4142914, 70.8457794, -86.9355316, 68.8919830, -158.3062744, 157.7812958
1: -74.2818604, 62.9273872, -72.2399521, 61.1490402, -135.4308929, 135.1673126
2: -97.9210205, 64.6900330, -95.1595993, 62.8797340, -160.8007202, 159.8496399
3: -104.2539825, 55.7844658, -101.2925491, 54.1809311, -158.4349060, 157.0769958
4: -95.2772827, 73.6762390, -92.6198959, 71.6562195, -166.9335022, 166.2961426
5: -85.9965820, 67.1936111, -83.6782227, 65.3583984, -151.3549805, 150.8718262
6: -82.4676666, 78.7554321, -80.1566772, 76.5163422, -158.9839935, 158.9120941
7: -89.5555801, 75.9351044, -87.0419998, 73.7957611, -163.3513489, 162.9770813
8: -107.6875534, 73.4866180, -104.6565552, 71.4415054, -179.1290436, 178.1431732
9: -81.7263718, 80.5646439, -79.3842239, 78.2599564, -159.9863281, 159.9488678

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4077733, upper bound: 189.4078460
time: 7.75 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4099113, upper bound: 189.4098586
time: 6.41 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -89.9342346, 71.2610931, -91.0210419, 72.1415634, -162.0758057, 162.2821198
1: -74.7221222, 63.3068314, -75.6889420, 64.0815811, -138.8036652, 138.9957733
2: -98.4990845, 65.0692749, -99.6818695, 65.8262100, -164.3252869, 164.7511444
3: -104.8794937, 56.1216393, -106.1462784, 56.7674599, -161.6469269, 162.2679138
4: -95.8379211, 74.1107559, -97.0149231, 75.0424576, -170.8803711, 171.1256714
5: -86.4968414, 67.5942230, -87.6104050, 68.4680634, -154.9648895, 155.2046051
6: -82.9561234, 79.2210312, -83.9544296, 80.1488266, -163.1049194, 163.1754608
7: -90.0894928, 76.3876953, -91.1924057, 77.3001480, -167.3896484, 167.5800934
8: -108.3217773, 73.9156036, -109.6069794, 74.7874908, -183.1092224, 183.5225677
9: -82.2215042, 81.0465546, -83.1953506, 81.9876328, -164.2091370, 164.2419128

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4078002, upper bound: 189.4078886
time: 7.18 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4098948, upper bound: 189.4098948
time: 7.08 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 16.60 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.60
Output dim: 7, lower bound: -189.4118914, upper bound: 189.4119851
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.60
Output dim: 7, lower bound: -189.4120178, upper bound: 189.4120472
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.60
Output dim: 7, lower bound: -189.4118914, upper bound: 189.4119852
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.60
Output dim: 7, lower bound: -189.4120178, upper bound: 189.4120472
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.60
Output dim: 7, lower bound: -189.4077733, upper bound: 189.4078460
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.60
Output dim: 7, lower bound: -189.4099113, upper bound: 189.4098586
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.60
Output dim: 7, lower bound: -189.4078002, upper bound: 189.4078886
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.60
Output dim: 7, lower bound: -189.4098948, upper bound: 189.4098948

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -82.1124496, 65.0851974, -90.8610001, 72.0018539, -154.1143036, 155.9461975
1: -68.1654358, 57.7319756, -75.5312653, 63.9704475, -132.1358643, 133.2632294
2: -89.8305283, 59.4538994, -99.5208359, 65.7123947, -155.5428925, 158.9747314
3: -95.6255112, 51.2016373, -105.9571609, 56.7024536, -152.3279572, 157.1587982
4: -87.4237137, 67.6768188, -96.8606186, 74.9079742, -162.3316650, 164.5374298
5: -79.0401917, 61.7562218, -87.4266052, 68.3163147, -147.3565063, 149.1828156
6: -75.7128677, 72.2533569, -83.8103485, 80.0344238, -155.7472534, 156.0636902
7: -82.1875305, 69.7568665, -91.0211105, 77.1522369, -159.3397675, 160.7779694
8: -98.8551025, 67.4950409, -109.4247894, 74.6989288, -173.5540314, 176.9198151
9: -74.9961853, 73.9275208, -83.0505753, 81.8719482, -156.8681183, 156.9780884

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4103478, upper bound: 189.4104745
time: 7.43 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4099026, upper bound: 189.4099663
time: 7.48 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -86.4389572, 68.5261459, -91.5181046, 72.5247879, -158.9637451, 160.0442505
1: -71.8221970, 60.8392029, -76.0883026, 64.4478226, -136.2700043, 136.9274902
2: -94.6204681, 62.5730591, -100.2518539, 66.1908875, -160.8113556, 162.8249207
3: -100.7705765, 53.9377708, -106.7459946, 57.1236420, -157.8942261, 160.6837616
4: -92.0837097, 71.2642899, -97.5714874, 75.4559326, -167.5395966, 168.8357849
5: -83.2030411, 65.0486526, -88.0572281, 68.8190231, -152.0220642, 153.1058807
6: -79.7374573, 76.1020279, -84.4265594, 80.6228790, -160.3603363, 160.5285797
7: -86.5888596, 73.4655533, -91.6966400, 77.7207413, -164.3096008, 165.1621704
8: -104.0988312, 71.0338898, -110.2255249, 75.2383652, -179.3371887, 181.2593536
9: -79.0325470, 77.8765335, -83.6726074, 82.4792099, -161.5117493, 161.5491180

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4104648, upper bound: 189.4105573
time: 7.99 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4100430, upper bound: 189.4100430
time: 7.72 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -82.1124496, 65.0851974, -89.4142914, 70.8457794, -152.9582214, 154.4994812
1: -68.1654358, 57.7319756, -74.2818604, 62.9273872, -131.0928192, 132.0138092
2: -89.8305283, 59.4538994, -97.9210205, 64.6900330, -154.5205231, 157.3749237
3: -95.6255112, 51.2016373, -104.2539825, 55.7844658, -151.4099731, 155.4556274
4: -87.4237137, 67.6768188, -95.2772827, 73.6762390, -161.0999451, 162.9541016
5: -79.0401917, 61.7562218, -85.9965820, 67.1936111, -146.2337952, 147.7528076
6: -75.7128677, 72.2533569, -82.4676666, 78.7554321, -154.4682465, 154.7209930
7: -82.1875305, 69.7568665, -89.5555801, 75.9351044, -158.1226196, 159.3124390
8: -98.8551025, 67.4950409, -107.6875534, 73.4866180, -172.3417206, 175.1825409
9: -74.9961853, 73.9275208, -81.7263718, 80.5646439, -155.5608215, 155.6539001

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4083274, upper bound: 189.4082954
time: 6.96 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4104107, upper bound: 189.4104940
time: 8.01 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -86.4389572, 68.5261459, -89.9342346, 71.2610931, -157.7000427, 158.4603882
1: -71.8221970, 60.8392029, -74.7221222, 63.3068314, -135.1289825, 135.5612793
2: -94.6204681, 62.5730591, -98.4990845, 65.0692749, -159.6897430, 161.0721436
3: -100.7705765, 53.9377708, -104.8794937, 56.1216393, -156.8922119, 158.8172302
4: -92.0837097, 71.2642899, -95.8379211, 74.1107559, -166.1944275, 167.1022034
5: -83.2030411, 65.0486526, -86.4968414, 67.5942230, -150.7972565, 151.5454865
6: -79.7374573, 76.1020279, -82.9561234, 79.2210312, -158.9584808, 159.0581512
7: -86.5888596, 73.4655533, -90.0894928, 76.3876953, -162.9765625, 163.5550385
8: -104.0988312, 71.0338898, -108.3217773, 73.9156036, -178.0144043, 179.3556366
9: -79.0325470, 77.8765335, -82.2215042, 81.0465546, -160.0791016, 160.0980377

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4083247, upper bound: 189.4083939
time: 8.51 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4105407, upper bound: 189.4105813
time: 7.21 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -84.4169846, 66.8814621, -85.0632019, 67.4085770, -151.8255615, 151.9446716
1: -70.0738068, 59.3768463, -70.6657028, 59.8230591, -129.8968658, 130.0425415
2: -92.3844681, 61.1450119, -93.0877457, 61.5534706, -153.9379120, 154.2327576
3: -98.3819275, 52.6930428, -99.0934296, 53.0253983, -151.4073181, 151.7864685
4: -89.8088150, 69.5327759, -90.5801239, 70.1089020, -159.9177094, 160.1128845
5: -81.1889267, 63.4093895, -81.8789368, 63.9470215, -145.1359100, 145.2882996
6: -77.8465805, 74.3120422, -78.4263535, 74.8528671, -152.6994324, 152.7383728
7: -84.4623184, 71.7167435, -85.1412811, 72.2175980, -156.6799011, 156.8580322
8: -101.6792297, 69.3818359, -102.4013138, 69.9043427, -171.5835724, 171.7831421
9: -77.1167679, 76.0285797, -77.6622391, 76.5640335, -153.6808014, 153.6908264

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 128

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4035461, upper bound: 189.4033131
time: 6.93 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4030490, upper bound: 189.4028391
time: 7.18 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -86.2901688, 68.3687134, -86.7831421, 68.7711487, -155.0613098, 155.1518555
1: -71.6487350, 60.7098198, -72.1115646, 61.0409317, -132.6896667, 132.8213806
2: -94.4618454, 62.4781189, -94.9908676, 62.7718697, -157.2336731, 157.4689941
3: -100.5866852, 53.8525314, -101.1137314, 54.0867386, -154.6734314, 154.9662476
4: -91.8785629, 71.0942230, -92.4541626, 71.5303192, -163.4088593, 163.5483551
5: -82.9972153, 64.8352509, -83.5319824, 65.2434998, -148.2407227, 148.3672180
6: -79.5832062, 75.9757462, -80.0159912, 76.3807297, -155.9639130, 155.9916992
7: -86.3839035, 73.3054199, -86.8872986, 73.6675491, -160.0514526, 160.1927185
8: -103.9198074, 70.9178238, -104.4726791, 71.3162155, -175.2359924, 175.3905029
9: -78.8574829, 77.7288742, -79.2443390, 78.1216812, -156.9791565, 156.9732056

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 249

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4057337, upper bound: 189.4056419
time: 8.12 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4052008, upper bound: 189.4051163
time: 6.93 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -84.9267273, 67.2888718, -89.1406860, 70.6522598, -155.5789795, 156.4295654
1: -70.5048904, 59.7494812, -74.1081696, 62.7507362, -133.2556305, 133.8576508
2: -92.9510803, 61.5172844, -97.6011505, 64.4939728, -157.4450531, 159.1184387
3: -98.9947815, 53.0242081, -103.9389801, 55.6079826, -154.6027679, 156.9631958
4: -90.3584976, 69.9590149, -94.9668121, 73.4888458, -163.8473358, 164.9257965
5: -81.6792603, 63.8030090, -85.8041153, 67.0509949, -148.7302399, 149.6071014
6: -78.3256607, 74.7685013, -82.2175293, 78.4782333, -156.8038940, 156.9860077
7: -84.9865112, 72.1613312, -89.2835159, 75.7155151, -160.7020264, 161.4448395
8: -102.3003006, 69.8027496, -107.3416061, 73.2445831, -175.5448914, 177.1443329
9: -77.6034470, 76.5019150, -81.4665298, 80.2849579, -157.8883972, 157.9684143

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 128

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4035168, upper bound: 189.4033744
time: 8.38 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4029778, upper bound: 189.4028948
time: 8.38 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -86.8143768, 68.7871780, -90.8697891, 72.0216599, -158.8359985, 159.6569519
1: -72.0927658, 61.0922356, -75.5615005, 63.9742050, -136.0669708, 136.6537323
2: -95.0446014, 62.8603401, -99.5143738, 65.7190704, -160.7636414, 162.3747101
3: -101.2170334, 54.1914444, -105.9686737, 56.6738815, -157.8909149, 160.1601257
4: -92.4437714, 71.5322266, -96.8503799, 74.9174881, -167.3612518, 168.3825989
5: -83.5011826, 65.2387772, -87.4652405, 68.3539963, -151.8551788, 152.7040100
6: -80.0750504, 76.4451294, -83.8147049, 80.0142212, -160.0892639, 160.2598267
7: -86.9221649, 73.7612000, -91.0388489, 77.1728439, -164.0950012, 164.8000488
8: -104.5588455, 71.3504333, -109.4244919, 74.6630402, -179.2218933, 180.7749329
9: -79.3564606, 78.2145081, -83.0564423, 81.8503418, -161.2067871, 161.2709503

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4056805, upper bound: 189.4056471
time: 7.98 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4051156, upper bound: 189.4051156
time: 6.87 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 17.18 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.18
Output dim: 7, lower bound: -189.4103478, upper bound: 189.4104745
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.18
Output dim: 7, lower bound: -189.4099026, upper bound: 189.4099663
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.18
Output dim: 7, lower bound: -189.4104648, upper bound: 189.4105573
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.18
Output dim: 7, lower bound: -189.4100430, upper bound: 189.4100430
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.18
Output dim: 7, lower bound: -189.4083274, upper bound: 189.4082954
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.18
Output dim: 7, lower bound: -189.4104107, upper bound: 189.4104940
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.18
Output dim: 7, lower bound: -189.4083247, upper bound: 189.4083939
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.18
Output dim: 7, lower bound: -189.4105407, upper bound: 189.4105813
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.18
Output dim: 7, lower bound: -189.4035461, upper bound: 189.4033131
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.18
Output dim: 7, lower bound: -189.4030490, upper bound: 189.4028391
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.18
Output dim: 7, lower bound: -189.4057337, upper bound: 189.4056419
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.18
Output dim: 7, lower bound: -189.4052008, upper bound: 189.4051163
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.18
Output dim: 7, lower bound: -189.4035168, upper bound: 189.4033744
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.18
Output dim: 7, lower bound: -189.4029778, upper bound: 189.4028948
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.18
Output dim: 7, lower bound: -189.4056805, upper bound: 189.4056471
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.18
Output dim: 7, lower bound: -189.4051156, upper bound: 189.4051156

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -80.7694016, 64.0242081, -83.2589035, 65.9976425, -146.7670441, 147.2831116
1: -67.0457611, 56.7913818, -69.1920166, 58.6450233, -125.6907806, 125.9833908
2: -88.3527679, 58.5113678, -91.1573639, 60.3744354, -148.7272034, 149.6687012
3: -94.0649567, 50.3824577, -97.1205826, 52.0614929, -146.1264496, 147.5030060
4: -85.9801483, 66.5734482, -88.6915588, 68.6638489, -154.6439819, 155.2650146
5: -77.7518845, 60.7574959, -80.1341705, 62.6623535, -140.4142456, 140.8916626
6: -74.4762878, 71.0716782, -76.8134079, 73.3445206, -147.8208008, 147.8850861
7: -80.8437195, 68.6393280, -83.4153595, 70.8215714, -151.6652832, 152.0546875
8: -97.2363510, 66.3866653, -100.2663651, 68.4323120, -165.6686554, 166.6530304
9: -73.7779617, 72.7288361, -76.1556168, 75.0861435, -148.8640747, 148.8844299

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 128

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4057068, upper bound: 189.4058251
time: 8.14 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4087067, upper bound: 189.4088319
time: 9.51 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -78.5251389, 62.2534943, -83.8013382, 66.4212112, -144.9463501, 146.0548248
1: -65.1699677, 55.2138062, -69.6220627, 58.9903526, -124.1603088, 124.8358688
2: -85.8809128, 56.9330750, -91.7176361, 60.7654343, -146.6463470, 148.6506958
3: -91.4464645, 48.9995613, -97.7304840, 52.3420868, -143.7885132, 146.7300262
4: -83.5649414, 64.7301102, -89.2615891, 69.0854416, -152.6503448, 153.9916382
5: -75.6007996, 59.0913582, -80.6474304, 63.0650444, -138.6658478, 139.7387848
6: -72.4063797, 69.0919571, -77.2795715, 73.8168182, -146.2232056, 146.3715057
7: -78.5999985, 66.7730942, -83.9400711, 71.2964554, -149.8964539, 150.7131653
8: -94.5245056, 64.5350037, -100.8778305, 68.8152466, -163.3397217, 165.4128113
9: -71.7446899, 70.7220306, -76.6730957, 75.5603638, -147.3050537, 147.3951263

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 128

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4053719, upper bound: 189.4054338
time: 8.56 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4082346, upper bound: 189.4083097
time: 6.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -85.0964432, 67.4657135, -83.8843079, 66.4952011, -151.5916138, 151.3500214
1: -70.7027893, 59.8983688, -69.7220078, 59.0992432, -129.8020325, 129.6203613
2: -93.1429138, 61.6303596, -91.8525162, 60.8295746, -153.9724884, 153.4828491
3: -99.2101593, 53.1180916, -97.8708420, 52.4622688, -151.6724091, 150.9889374
4: -90.6410217, 70.1615295, -89.3676758, 69.1850967, -159.8261108, 159.5292053
5: -81.9153366, 64.0503159, -80.7339859, 63.1410065, -145.0563354, 144.7843018
6: -78.5013885, 74.9203415, -77.3991470, 73.9046707, -152.4060516, 152.3194885
7: -85.2457047, 72.3480148, -84.0584412, 71.3630524, -156.6087646, 156.4064636
8: -102.4810562, 69.9262924, -101.0281906, 68.9457550, -171.4268188, 170.9544830
9: -77.8146439, 76.6783142, -76.7490540, 75.6641922, -153.4788361, 153.4273529

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 128

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4058044, upper bound: 189.4059484
time: 7.59 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4088413, upper bound: 189.4089177
time: 9.16 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -82.8479385, 65.6920547, -84.4397430, 66.9307175, -149.7786560, 150.1318054
1: -68.8244781, 58.3193054, -70.1633530, 59.4549599, -128.2794342, 128.4826660
2: -90.6674881, 60.0498390, -92.4275970, 61.2304993, -151.8979797, 152.4774323
3: -96.5880356, 51.7340889, -98.4967575, 52.7522125, -149.3402405, 150.2308502
4: -88.2225342, 68.3140564, -89.9521561, 69.6196747, -157.8421936, 158.2662048
5: -79.7601471, 62.3823509, -81.2609940, 63.5556374, -143.3157654, 143.6433411
6: -76.4270401, 72.9383698, -77.8792038, 74.3891296, -150.8161621, 150.8175354
7: -82.9987411, 70.4788055, -84.5972672, 71.8501740, -154.8489075, 155.0760803
8: -99.7629700, 68.0694199, -101.6567535, 69.3403473, -169.1033173, 169.7261505
9: -75.7774582, 74.6681595, -77.2793427, 76.1515961, -151.9289856, 151.9475098

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 128

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4054753, upper bound: 189.4055585
time: 7.98 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4083981, upper bound: 189.4083981
time: 6.77 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -80.1997757, 63.5692863, -84.4169846, 66.8814621, -147.0812378, 147.9862671
1: -66.5570145, 56.3770638, -70.0738068, 59.3768463, -125.9338608, 126.4508667
2: -87.7129593, 58.0984268, -92.3844681, 61.1450119, -148.8579407, 150.4828949
3: -93.3785706, 50.0209961, -98.3819275, 52.6930428, -146.0716095, 148.4029083
4: -85.3395233, 66.0955658, -89.8088150, 69.5327759, -154.8722839, 155.9043732
5: -77.2018967, 60.3136330, -81.1889267, 63.4093895, -140.6112823, 141.5025635
6: -73.9455338, 70.5532761, -77.8465805, 74.3120422, -148.2575684, 148.3998566
7: -80.2440338, 68.1438065, -84.4623184, 71.7167435, -151.9607849, 152.6061249
8: -96.5508499, 65.9250107, -101.6792297, 69.3818359, -165.9326782, 167.6042175
9: -73.2357712, 72.1936035, -77.1167679, 76.0285797, -149.2643433, 149.3103638

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 128

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4038084, upper bound: 189.4041159
time: 8.64 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4033642, upper bound: 189.4036765
time: 7.42 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -81.9637909, 64.9673233, -86.2901688, 68.3687134, -150.3324738, 151.2574921
1: -68.0401993, 57.6264954, -71.6487350, 60.7098198, -128.7500153, 129.2752075
2: -89.6659241, 59.3487053, -94.4618454, 62.4781189, -152.1440430, 153.8105316
3: -95.4511108, 51.1097603, -100.5866852, 53.8525314, -149.3035889, 151.6964417
4: -87.2620087, 67.5539856, -91.8785629, 71.0942230, -158.3562164, 159.4325409
5: -78.8975296, 61.6441498, -82.9972153, 64.8352509, -143.7327423, 144.6413574
6: -75.5756302, 72.1211090, -79.5832062, 75.9757462, -151.5513611, 151.7042999
7: -82.0367661, 69.6318970, -86.3839035, 73.3054199, -155.3421783, 156.0157928
8: -98.6757584, 67.3727264, -103.9198074, 70.9178238, -169.5935822, 171.2925415
9: -74.8597641, 73.7926941, -78.8574829, 77.7288742, -152.5886383, 152.6501770

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 128

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4062040, upper bound: 189.4062733
time: 8.49 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4057375, upper bound: 189.4058523
time: 8.05 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -84.5121841, 67.0000076, -84.9267273, 67.2888718, -151.8010559, 151.9267273
1: -70.2020340, 59.4749947, -70.5048904, 59.7494812, -129.9514771, 129.9798889
2: -92.4878922, 61.2077637, -92.9510803, 61.5172844, -154.0051270, 154.1588287
3: -98.5080185, 52.7493858, -98.9947815, 53.0242081, -151.5322266, 151.7441711
4: -89.9845428, 69.6715775, -90.3584976, 69.9590149, -159.9435425, 160.0300751
5: -81.3522339, 63.5963860, -81.6792603, 63.8030090, -145.1552277, 145.2756348
6: -77.9573288, 74.3898392, -78.3256607, 74.7685013, -152.7258301, 152.7154999
7: -84.6317520, 71.8417511, -84.9865112, 72.1613312, -156.7930908, 156.8282623
8: -101.7778244, 69.4525909, -102.3003006, 69.8027496, -171.5805511, 171.7528992
9: -77.2601852, 76.1307755, -77.6034470, 76.5019150, -153.7621002, 153.7342224

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 249

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4039468, upper bound: 189.4041570
time: 8.15 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4033642, upper bound: 189.4037444
time: 8.28 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -86.2917252, 68.4093857, -86.8143768, 68.7871780, -155.0789032, 155.2237396
1: -71.6981506, 60.7346802, -72.0927658, 61.0922356, -132.7903442, 132.8274384
2: -94.4573975, 62.4687920, -95.0446014, 62.8603401, -157.3177338, 157.5133820
3: -100.5977097, 53.8466644, -101.2170334, 54.1914444, -154.7891541, 155.0636902
4: -91.9235458, 71.1426392, -92.4437714, 71.5322266, -163.4557800, 163.5863800
5: -83.0617218, 64.9376144, -83.5011826, 65.2387772, -148.3005066, 148.4387970
6: -79.6015015, 75.9709930, -80.0750504, 76.4451294, -156.0466309, 156.0460205
7: -86.4394455, 73.3416977, -86.9221649, 73.7612000, -160.2006531, 160.2638550
8: -103.9211731, 70.9127274, -104.5588455, 71.3504333, -175.2716064, 175.4715729
9: -78.8973999, 77.7428970, -79.3564606, 78.2145081, -157.1119080, 157.0993500

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 128

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4062633, upper bound: 189.4062901
time: 9.46 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4058202, upper bound: 189.4058570
time: 8.13 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -83.0756302, 65.8222885, -77.5070572, 61.4402084, -144.5158386, 143.3293457
1: -68.9560776, 58.4373207, -64.3650208, 54.5307693, -123.4868393, 122.8023376
2: -90.9084473, 60.2030334, -84.7724609, 56.2491188, -147.1575165, 144.9754791
3: -96.8234711, 51.8744011, -90.3120270, 48.4133110, -145.2367859, 142.1864319
4: -88.3675232, 68.4312668, -82.4601212, 63.9033241, -152.2708435, 150.8913879
5: -79.9024734, 62.4119225, -74.6312180, 58.3289185, -138.2313843, 137.0431366
6: -76.6120605, 73.1320572, -71.4727478, 68.2040482, -144.8161011, 144.6047821
7: -83.1202316, 70.6001282, -77.5821686, 65.9296494, -149.0498810, 148.1822968
8: -100.0628738, 68.2757721, -93.2990036, 63.6757507, -163.7386169, 161.5747528
9: -75.9003372, 74.8316422, -70.8118668, 69.8207855, -145.7210999, 145.6435089

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 249

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4035461, upper bound: 189.4033104
time: 9.26 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4035461, upper bound: 189.4033131
time: 7.31 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -80.8607407, 64.0733643, -77.9922028, 61.8185577, -142.6792908, 142.0655670
1: -67.1027832, 56.8808861, -64.7446442, 54.8345833, -121.9373627, 121.6255341
2: -88.4698868, 58.6478615, -85.2709808, 56.6015854, -145.0714722, 143.9188385
3: -94.2381973, 50.5106163, -90.8516846, 48.6608963, -142.8990936, 141.3622589
4: -85.9812241, 66.6098404, -82.9659348, 64.2758331, -150.2570496, 149.5757751
5: -77.7770081, 60.7666283, -75.0870514, 58.6870422, -136.4640198, 135.8536682
6: -74.5679169, 71.1785431, -71.8854752, 68.6242676, -143.1921844, 143.0639954
7: -80.9071503, 68.7595139, -78.0500946, 66.3565445, -147.2637024, 146.8096008
8: -97.3866043, 66.4457397, -93.8401489, 64.0099716, -161.3965759, 160.2858887
9: -73.8931122, 72.8502426, -71.2757263, 70.2416000, -144.1347046, 144.1259460

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 249

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4030490, upper bound: 189.4028377
time: 6.49 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4030490, upper bound: 189.4028391
time: 7.72 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -84.9431610, 67.3047028, -79.2133865, 62.7927017, -147.7358398, 146.5180969
1: -70.5262527, 59.7659798, -65.8001785, 55.7394905, -126.2657318, 125.5661545
2: -92.9791336, 61.5318336, -86.6618652, 57.4582748, -150.4374084, 148.1936951
3: -99.0215607, 53.0300522, -92.3171997, 49.4666862, -148.4882355, 145.3472443
4: -90.4310303, 69.9879074, -84.3203659, 65.3142471, -155.7452698, 154.3082733
5: -81.7052536, 63.8332481, -76.2718353, 59.6157455, -141.3209839, 140.1050873
6: -78.3428116, 74.7905197, -73.0503006, 69.7207947, -148.0635986, 147.8408203
7: -85.0354843, 72.1838455, -79.3151855, 67.3683929, -152.4038391, 151.4990082
8: -102.2959213, 69.8065491, -95.3549957, 65.0768204, -167.3727417, 165.1615448
9: -77.6353073, 76.5266495, -72.3819580, 71.3671188, -149.0024261, 148.9085999

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 249

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4057304, upper bound: 189.4056419
time: 6.35 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4057304, upper bound: 189.4056419
time: 7.52 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -82.7243958, 65.5528717, -79.7001724, 63.1724968, -145.8968964, 145.2530518
1: -68.6700134, 58.2072334, -66.1807098, 56.0445404, -124.7145386, 124.3879318
2: -90.5362244, 59.9749146, -87.1621780, 57.8124466, -148.3486481, 147.1370850
3: -96.4319534, 51.6643791, -92.8585510, 49.7153282, -146.1472626, 144.5229187
4: -88.0414886, 68.1639786, -84.8283005, 65.6879272, -153.7294006, 152.9922791
5: -79.5765152, 62.1857491, -76.7289352, 59.9751129, -139.5516205, 138.9146881
6: -76.2963638, 72.8335953, -73.4644775, 70.1423416, -146.4387054, 146.2980652
7: -82.8203506, 70.3404312, -79.7851257, 67.7970581, -150.6174011, 150.1255493
8: -99.6156769, 67.9744415, -95.8986053, 65.4126663, -165.0283356, 163.8730316
9: -75.6267929, 74.5429840, -72.8474960, 71.7899170, -147.4166870, 147.3904724

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 249

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4051972, upper bound: 189.4051163
time: 6.83 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4051972, upper bound: 189.4051163
time: 6.77 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -83.5832062, 66.2274170, -81.4821396, 64.6026840, -148.1858826, 147.7095642
1: -69.3848572, 58.8080330, -67.7222290, 57.3829994, -126.7678528, 126.5302582
2: -91.4720230, 60.5734978, -89.1707764, 59.1125107, -150.5845337, 149.7442780
3: -97.4333496, 52.2037811, -95.0352325, 50.9281273, -148.3614807, 147.2389832
4: -88.9145126, 68.8552628, -86.7387772, 67.1978455, -156.1123657, 155.5940247
5: -80.3902664, 62.8034019, -78.4583817, 61.3535309, -141.7438049, 141.2617798
6: -77.0885468, 73.5861359, -75.1653519, 71.7369156, -148.8254547, 148.7514954
7: -83.6418839, 71.0424728, -81.6189575, 69.3364334, -152.9783173, 152.6614380
8: -100.6807556, 68.6945038, -98.1102295, 66.9278717, -167.6086273, 166.8047333
9: -76.3845444, 75.3026505, -74.5180054, 73.4466095, -149.8311462, 149.8206482

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 249

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4034686, upper bound: 189.4033299
time: 6.78 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4034686, upper bound: 189.4033726
time: 8.14 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -81.3816833, 64.4896927, -82.0885925, 65.0808258, -146.4624939, 146.5782776
1: -67.5439911, 57.2618752, -68.2061157, 57.7764549, -125.3204422, 125.4679871
2: -89.0483322, 59.0287437, -89.8049240, 59.5561333, -148.6044464, 148.8336639
3: -94.8654633, 50.8498344, -95.7216339, 51.2525291, -146.1179962, 146.5714722
4: -86.5426254, 67.0463409, -87.3731537, 67.6771088, -154.2197266, 154.4194946
5: -78.2783432, 61.1690407, -79.0331650, 61.8077202, -140.0860443, 140.2021790
6: -75.0580292, 71.6450806, -75.6967392, 72.2669296, -147.3249512, 147.3418274
7: -81.4429245, 69.2145233, -82.2148514, 69.8721237, -151.3150482, 151.4293823
8: -98.0209045, 66.8751450, -98.8006668, 67.3646088, -165.3855133, 165.6757812
9: -74.3903885, 73.3346863, -75.0986404, 73.9825821, -148.3729553, 148.4333191

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 249

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4029570, upper bound: 189.4028569
time: 8.04 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4029570, upper bound: 189.4028855
time: 8.42 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -85.4640198, 67.7205734, -83.2021255, 65.9646378, -151.4286499, 150.9226990
1: -70.9671402, 60.1459808, -69.1673279, 58.5996399, -129.5667725, 129.3132935
2: -93.5584412, 61.9118767, -91.0734253, 60.3308640, -153.8893127, 152.9853058
3: -99.6477585, 53.3668976, -97.0540390, 51.9881744, -151.6359100, 150.4209290
4: -90.9927063, 70.4231186, -88.6123505, 68.6192245, -159.6119385, 159.0354462
5: -82.2059708, 64.2341385, -80.1107101, 62.6492233, -144.8551788, 144.3448486
6: -78.8316422, 75.2568817, -76.7535629, 73.2645035, -152.0961304, 152.0104370
7: -85.5705872, 72.6367264, -83.3645248, 70.7851944, -156.3557739, 156.0012512
8: -102.9309845, 70.2363358, -100.1815796, 68.3388596, -171.2698364, 170.4179077
9: -78.1309967, 77.0093765, -76.0995026, 75.0036545, -153.1346436, 153.1088867

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4056542, upper bound: 189.4056471
time: 6.55 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4056542, upper bound: 189.4056323
time: 6.05 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -83.2583771, 65.9795227, -83.8070679, 66.4421082, -149.7004700, 149.7865906
1: -69.1226959, 58.5974693, -69.6502838, 58.9928856, -128.1155853, 128.2477570
2: -91.1302032, 60.3651428, -91.7071152, 60.7739449, -151.9041443, 152.0722656
3: -97.0746078, 52.0106239, -97.7392654, 52.3125000, -149.3871155, 149.7498932
4: -88.6178894, 68.6110382, -89.2458878, 69.0974045, -157.7152863, 157.8569183
5: -80.0902557, 62.5974998, -80.6842346, 63.1030731, -143.1933289, 143.2816925
6: -76.7983856, 73.3120422, -77.2846832, 73.7937317, -150.5921173, 150.5967102
7: -83.3695145, 70.8058701, -83.9595490, 71.3202591, -154.6897736, 154.7653961
8: -100.2668381, 68.4150925, -100.8715439, 68.7753448, -169.0421753, 169.2866364
9: -76.1359787, 75.0392685, -76.6792145, 75.5390167, -151.6749725, 151.7184601

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 249

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4050824, upper bound: 189.4051156
time: 6.86 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4050824, upper bound: 189.4050874
time: 7.44 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 16.61 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4057068, upper bound: 189.4058251
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4087067, upper bound: 189.4088319
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4053719, upper bound: 189.4054338
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4082346, upper bound: 189.4083097
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4058044, upper bound: 189.4059484
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4088413, upper bound: 189.4089177
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4054753, upper bound: 189.4055585
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4083981, upper bound: 189.4083981
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4038084, upper bound: 189.4041159
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4033642, upper bound: 189.4036765
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4062040, upper bound: 189.4062733
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4057375, upper bound: 189.4058523
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4039468, upper bound: 189.4041570
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4033642, upper bound: 189.4037444
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4062633, upper bound: 189.4062901
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4058202, upper bound: 189.4058570
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4035461, upper bound: 189.4033104
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4035461, upper bound: 189.4033131
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4030490, upper bound: 189.4028377
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4030490, upper bound: 189.4028391
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4057304, upper bound: 189.4056419
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4057304, upper bound: 189.4056419
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4051972, upper bound: 189.4051163
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4051972, upper bound: 189.4051163
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4034686, upper bound: 189.4033299
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4034686, upper bound: 189.4033726
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4029570, upper bound: 189.4028569
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4029570, upper bound: 189.4028855
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4056542, upper bound: 189.4056471
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4056542, upper bound: 189.4056323
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4050824, upper bound: 189.4051156
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.61
Output dim: 7, lower bound: -189.4050824, upper bound: 189.4050874

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -75.3425446, 59.7174683, -81.3444138, 64.4802628, -139.8228149, 141.0618744
1: -62.4751701, 52.9356194, -67.5817490, 57.2889595, -119.7641144, 120.5173569
2: -82.3348465, 54.6624947, -89.0373383, 59.0173569, -141.3522034, 143.6998138
3: -87.6837845, 47.0241508, -94.8716278, 50.8800316, -138.5637970, 141.8957672
4: -80.0459366, 62.0752220, -86.6047745, 67.0805206, -147.1264496, 148.6799927
5: -72.5293579, 56.6475182, -78.2941513, 61.2182884, -133.7476501, 134.9416504
6: -69.4557724, 66.2423477, -75.0440216, 71.6425400, -141.0983124, 141.2863312
7: -75.3104706, 64.0540619, -81.4695663, 69.2070923, -144.5175629, 145.5236206
8: -90.7033920, 61.9259644, -97.9594498, 66.8607483, -157.5641327, 159.8854065
9: -68.7670746, 67.7985077, -74.3933411, 73.3503876, -142.1174469, 142.1918488

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 249

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4057027, upper bound: 189.4058064
time: 8.15 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4057027, upper bound: 189.4058251
time: 7.08 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -77.7647858, 61.6417999, -83.1099625, 65.8795166, -143.6443024, 144.7517395
1: -64.5134277, 54.6593933, -69.0664978, 58.5393372, -123.0527649, 123.7258759
2: -85.0259705, 56.3848381, -90.9924240, 60.2690811, -145.2950439, 147.3772583
3: -90.5401077, 48.5256271, -96.9458008, 51.9694519, -142.5095215, 145.4714355
4: -82.7120590, 64.0909195, -88.5295792, 68.5408173, -151.2528687, 152.6204987
5: -74.8687134, 58.4928932, -79.9911957, 62.5500946, -137.4188080, 138.4840698
6: -71.7028656, 68.3977737, -76.6759491, 73.2119980, -144.9148560, 145.0737152
7: -77.7970657, 66.1137085, -83.2643204, 70.6963806, -148.4934387, 149.3779907
8: -93.6115875, 63.9139214, -100.0867081, 68.3097839, -161.9213562, 164.0006104
9: -71.0220947, 70.0025787, -76.0190201, 74.9510269, -145.9731140, 146.0215912

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 249

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4056819, upper bound: 189.4057629
time: 9.07 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4056819, upper bound: 189.4088319
time: 8.39 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -73.0999680, 57.9472198, -81.8895645, 64.9065857, -138.0065613, 139.8367462
1: -60.6008911, 51.3583450, -68.0149002, 57.6362801, -118.2371674, 119.3732376
2: -79.8632050, 53.0849838, -89.6010742, 59.4102592, -139.2734680, 142.6860657
3: -85.0664368, 45.6420403, -95.4854202, 51.1624794, -136.2289124, 141.1274414
4: -77.6315765, 60.2319756, -87.1783218, 67.5052490, -145.1367950, 147.4102936
5: -70.3795776, 54.9815979, -78.8106918, 61.6232758, -132.0028229, 133.7922668
6: -67.3865662, 64.2636795, -75.5132751, 72.1179581, -139.5045166, 139.7769470
7: -73.0674286, 62.1886330, -81.9976196, 69.6846466, -142.7520294, 144.1862488
8: -87.9925690, 60.0742645, -98.5741348, 67.2463913, -155.2389221, 158.6483765
9: -66.7340088, 65.7920456, -74.9140244, 73.8281174, -140.5621185, 140.7060547

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4053510, upper bound: 189.4053783
time: 7.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4053510, upper bound: 189.4054338
time: 8.01 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 12.82 + 600.63 = 613.45 seconds
