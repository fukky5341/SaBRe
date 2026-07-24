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
execution time: IAR + RelationalAnalysis = 0.84 + 10.48 = 11.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -189.4203333, upper bound: 189.4203333

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4038513, upper bound: 189.4021679
time: 8.05 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4201938, upper bound: 189.4201938
time: 6.87 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 15.00 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 15.00
Output dim: 7, lower bound: -189.4038513, upper bound: 189.4021679
NS_A2, status: Status.UNKNOWN, split count: 1, time: 15.00
Output dim: 7, lower bound: -189.4201938, upper bound: 189.4201938

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -97.3713303, 77.1248703, -101.8211288, 80.6571884, -178.0285187, 178.9459991
1: -80.9806519, 68.5304642, -84.7619781, 71.7289658, -152.7096252, 153.2924500
2: -106.6730042, 70.2953033, -111.6222153, 73.4980774, -180.1710815, 181.9175110
3: -113.5506744, 60.6859016, -118.8323822, 63.4895363, -177.0402069, 179.5182495
4: -103.7944183, 80.1976852, -108.6636963, 83.9425430, -187.7369537, 188.8613892
5: -93.6626816, 73.0850677, -97.9635162, 76.5017395, -170.1644135, 171.0485840
6: -89.7819748, 85.7568817, -93.9107590, 89.7265854, -179.5085602, 179.6676331
7: -97.4864960, 82.5338364, -102.0390167, 86.3396835, -183.8261414, 184.5728455
8: -117.3117828, 80.0152588, -122.6215363, 83.6883926, -201.0001526, 202.6367798
9: -88.8598785, 87.6405563, -93.0304489, 91.7288513, -180.5887299, 180.6710052

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 159

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3986063, upper bound: 189.3963211
time: 7.22 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4022219, upper bound: 189.4004230
time: 8.55 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -102.4326706, 81.1419983, -103.0916901, 81.6635971, -184.0962677, 184.2336884
1: -85.2770920, 72.1630249, -85.8328323, 72.6308670, -157.9079285, 157.9958496
2: -112.2985458, 73.9368057, -113.0277863, 74.4069977, -186.7055359, 186.9645844
3: -119.5538712, 63.8698807, -120.3313828, 64.2784958, -183.8323669, 184.2012634
4: -109.3266907, 84.4495926, -110.0417786, 84.9963455, -194.3230133, 194.4913635
5: -98.5510178, 76.9643402, -99.1850357, 77.4622269, -176.0132446, 176.1493530
6: -94.4766769, 90.2701187, -95.0854187, 90.8557663, -185.3324280, 185.3555298
7: -102.6582184, 86.8594971, -103.3243332, 87.4172974, -190.0755005, 190.1838074
8: -123.3582306, 84.1924210, -124.1503143, 84.7329102, -208.0911407, 208.3427124
9: -93.5967941, 92.2854080, -94.2049332, 92.8833542, -186.4801483, 186.4903412

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3914398, upper bound: 189.3880788
time: 8.60 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3824246, upper bound: 189.3824245
time: 5.24 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 14.67 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.67
Output dim: 7, lower bound: -189.3986063, upper bound: 189.3963211
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.67
Output dim: 7, lower bound: -189.4022219, upper bound: 189.4004230
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.67
Output dim: 7, lower bound: -189.3914398, upper bound: 189.3880788
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.67
Output dim: 7, lower bound: -189.3824246, upper bound: 189.3824245

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -95.4563370, 75.6072083, -96.3203735, 76.2920227, -171.7483521, 171.9275818
1: -79.3700485, 67.1744232, -80.1283188, 67.8223953, -147.1924438, 147.3027344
2: -104.5537186, 68.9370728, -105.5241928, 69.5947723, -174.1484833, 174.4612579
3: -111.3020554, 59.5042000, -112.3643799, 60.0870018, -171.3890533, 171.8685760
4: -101.7074432, 78.6151276, -102.6489716, 79.3851242, -181.0925140, 181.2640991
5: -91.8221970, 71.6410599, -92.6696167, 72.3375320, -164.1597290, 164.3106689
6: -88.0125504, 84.0551147, -88.8198776, 84.8316422, -172.8441925, 172.8750000
7: -95.5414505, 80.9181595, -96.4310989, 81.6905289, -177.2319641, 177.3492584
8: -115.0046234, 78.4437408, -115.9940948, 79.1683960, -194.1729889, 194.4377747
9: -87.0968323, 85.9054489, -87.9521255, 86.7318420, -173.8286743, 173.8575745

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3924103, upper bound: 189.3890840
time: 8.89 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3924103, upper bound: 189.3963211
time: 8.84 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -97.2243195, 77.0083389, -98.7591019, 78.2283325, -175.4526215, 175.7674408
1: -80.8568726, 68.4261627, -82.1810608, 69.5567017, -150.4135742, 150.6072083
2: -106.5102539, 70.1912689, -108.2314148, 71.3305359, -177.8407898, 178.4226837
3: -113.3781738, 60.5950394, -115.2380676, 61.5958595, -174.9740295, 175.8330688
4: -103.6345215, 80.0762482, -105.3330078, 81.4130402, -185.0475311, 185.4092560
5: -93.5216064, 72.9742432, -95.0234299, 74.1934433, -167.7150574, 167.9976501
6: -89.6462555, 85.6261139, -91.0826874, 87.0018539, -176.6481018, 176.7088013
7: -97.3373489, 82.4102249, -98.9328079, 83.7633286, -181.1006470, 181.3430328
8: -117.1344986, 79.8943558, -118.9255676, 81.1684036, -198.3028870, 198.8199158
9: -88.7250137, 87.5071945, -90.2207184, 88.9506607, -177.6756439, 177.7279053

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 159

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3928990, upper bound: 189.3896711
time: 7.94 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3928990, upper bound: 189.4004230
time: 7.01 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -102.4326706, 81.1419983, -102.6474304, 81.3115311, -183.7442017, 183.7894135
1: -85.2770920, 72.1630249, -85.4586639, 72.3173904, -157.5944519, 157.6216888
2: -112.2985458, 73.9368057, -112.5371933, 74.0938339, -186.3923645, 186.4739838
3: -119.5538712, 63.8698807, -119.8119049, 64.0054321, -183.5592957, 183.6817627
4: -109.3266907, 84.4495926, -109.5619583, 84.6295929, -193.9562836, 194.0115509
5: -98.5510178, 76.9643402, -98.7571793, 77.1302032, -175.6811829, 175.7214966
6: -94.4766769, 90.2701187, -94.6757965, 90.4617462, -184.9384155, 184.9459229
7: -102.6582184, 86.8594971, -102.8790894, 87.0461349, -189.7043304, 189.7385864
8: -123.3582306, 84.1924210, -123.6131821, 84.3671341, -207.7253265, 207.8056030
9: -93.5967941, 92.2854080, -93.8014526, 92.4836197, -186.0803986, 186.0868530

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 155

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3840101, upper bound: 189.3807180
time: 9.57 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3839482, upper bound: 189.3806934
time: 7.17 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -101.4453201, 80.3593369, -91.6827011, 72.6037750, -174.0490875, 172.0420380
1: -84.4469299, 71.4677734, -76.2051849, 64.5625610, -149.0094757, 147.6729431
2: -111.2091370, 73.2419662, -100.4302673, 66.3922653, -177.6014099, 173.6722260
3: -118.4012527, 63.2648659, -107.0640106, 57.2560120, -175.6572418, 170.3288269
4: -108.2597275, 83.6352005, -97.6903305, 75.5128403, -183.7725677, 181.3255310
5: -97.5998764, 76.2264938, -88.1595306, 68.8911057, -166.4909821, 164.3859863
6: -93.5671463, 89.3947906, -84.5645676, 80.7399216, -174.3070679, 173.9593506
7: -101.6702042, 86.0362473, -91.9024887, 77.9275894, -179.5977936, 177.9387360
8: -122.1651001, 83.3789597, -110.3649368, 75.2659760, -197.4310760, 193.7438965
9: -92.7019653, 91.3980484, -83.8823166, 82.6267319, -175.3287048, 175.2803497

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 155

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3740715, upper bound: 189.3739072
time: 6.03 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3750171, upper bound: 189.3750171
time: 5.04 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 11.97 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 11.97
Output dim: 7, lower bound: -189.3924103, upper bound: 189.3890840
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 11.97
Output dim: 7, lower bound: -189.3924103, upper bound: 189.3963211
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 11.97
Output dim: 7, lower bound: -189.3928990, upper bound: 189.3896711
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 11.97
Output dim: 7, lower bound: -189.3928990, upper bound: 189.4004230
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 11.97
Output dim: 7, lower bound: -189.3840101, upper bound: 189.3807180
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 11.97
Output dim: 7, lower bound: -189.3839482, upper bound: 189.3806934
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 11.97
Output dim: 7, lower bound: -189.3740715, upper bound: 189.3739072
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 11.97
Output dim: 7, lower bound: -189.3750171, upper bound: 189.3750171

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -91.6852417, 72.6138763, -96.3203735, 76.2920227, -167.9772644, 168.9342499
1: -76.1902466, 64.4945984, -80.1283188, 67.8223953, -144.0126190, 144.6229248
2: -100.3727646, 66.2584686, -105.5241928, 69.5947723, -169.9675140, 171.7826538
3: -106.8693161, 57.1694984, -112.3643799, 60.0870018, -166.9563141, 169.5338440
4: -97.5817566, 75.4867401, -102.6489716, 79.3851242, -176.9668732, 178.1357117
5: -88.1910172, 68.7845535, -92.6696167, 72.3375320, -160.5285492, 161.4541626
6: -84.5214920, 80.6980820, -88.8198776, 84.8316422, -169.3531342, 169.5178833
7: -91.6946869, 77.7288208, -96.4310989, 81.6905289, -173.3852081, 174.1599121
8: -110.4609070, 75.3422089, -115.9940948, 79.1683960, -189.6292877, 191.3362427
9: -83.6103363, 82.4798355, -87.9521255, 86.7318420, -170.3421783, 170.4319611

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3896863, upper bound: 189.3872185
time: 7.66 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3920735, upper bound: 189.3889863
time: 7.61 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -94.3987808, 74.7683334, -96.3203735, 76.2920227, -170.6907959, 171.0886993
1: -78.4771194, 66.4220963, -80.1283188, 67.8223953, -146.2994995, 146.5504150
2: -103.3820114, 68.1917877, -105.5241928, 69.5947723, -172.9767303, 173.7159576
3: -110.0626297, 58.8486404, -112.3643799, 60.0870018, -170.1496277, 171.2130127
4: -100.5612793, 77.7425766, -102.6489716, 79.3851242, -179.9463654, 180.3915405
5: -90.8103638, 70.8442764, -92.6696167, 72.3375320, -163.1478882, 163.5138855
6: -87.0381165, 83.1127014, -88.8198776, 84.8316422, -171.8697510, 171.9325714
7: -94.4716339, 80.0343933, -96.4310989, 81.6905289, -176.1621704, 176.4654846
8: -113.7269058, 77.5702057, -115.9940948, 79.1683960, -192.8952789, 193.5642548
9: -86.1331635, 84.9439545, -87.9521255, 86.7318420, -172.8650055, 172.8960876

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3896863, upper bound: 189.3872185
time: 7.31 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3920735, upper bound: 189.3961782
time: 8.20 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -91.6852417, 72.6138763, -98.7591019, 78.2283325, -169.9135742, 171.3729858
1: -76.1902466, 64.4945984, -82.1810608, 69.5567017, -145.7469330, 146.6756592
2: -100.3727646, 66.2584686, -108.2314148, 71.3305359, -171.7033081, 174.4898834
3: -106.8693161, 57.1694984, -115.2380676, 61.5958595, -168.4651794, 172.4075317
4: -97.5817566, 75.4867401, -105.3330078, 81.4130402, -178.9947815, 180.8197479
5: -88.1910172, 68.7845535, -95.0234299, 74.1934433, -162.3844604, 163.8079529
6: -84.5214920, 80.6980820, -91.0826874, 87.0018539, -171.5233459, 171.7807312
7: -91.6946869, 77.7288208, -98.9328079, 83.7633286, -175.4579926, 176.6616211
8: -110.4609070, 75.3422089, -118.9255676, 81.1684036, -191.6293030, 194.2677460
9: -83.6103363, 82.4798355, -90.2207184, 88.9506607, -172.5609894, 172.7005615

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 159

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3896863, upper bound: 189.3877364
time: 7.81 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3920735, upper bound: 189.3895979
time: 8.61 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -94.3987808, 74.7683334, -98.7591019, 78.2283325, -172.6271057, 173.5274353
1: -78.4771194, 66.4220963, -82.1810608, 69.5567017, -148.0338135, 148.6031494
2: -103.3820114, 68.1917877, -108.2314148, 71.3305359, -174.7125397, 176.4232025
3: -110.0626297, 58.8486404, -115.2380676, 61.5958595, -171.6584930, 174.0867004
4: -100.5612793, 77.7425766, -105.3330078, 81.4130402, -181.9742889, 183.0755920
5: -90.8103638, 70.8442764, -95.0234299, 74.1934433, -165.0038147, 165.8676758
6: -87.0381165, 83.1127014, -91.0826874, 87.0018539, -174.0399628, 174.1953888
7: -94.4716339, 80.0343933, -98.9328079, 83.7633286, -178.2349548, 178.9671936
8: -113.7269058, 77.5702057, -118.9255676, 81.1684036, -194.8953094, 196.4957428
9: -86.1331635, 84.9439545, -90.2207184, 88.9506607, -175.0838165, 175.1646729

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 159

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3896863, upper bound: 189.3979019
time: 8.26 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3920735, upper bound: 189.4003660
time: 7.57 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -102.4326706, 81.1419983, -93.2918777, 73.9240952, -176.3567657, 174.4338684
1: -85.2770920, 72.1630249, -77.5788803, 65.7078934, -150.9849701, 149.7419128
2: -112.2985458, 73.9368057, -102.2167511, 67.4564590, -179.7550049, 176.1535645
3: -119.5538712, 63.8698807, -108.8348618, 58.2370224, -177.7908936, 172.7046967
4: -109.3266907, 84.4495926, -99.4836807, 76.9181747, -186.2448578, 183.9332733
5: -98.5510178, 76.9643402, -89.7558594, 70.1411133, -168.6921387, 166.7201538
6: -94.4766769, 90.2701187, -86.0633392, 82.1985931, -176.6752472, 176.3334656
7: -102.6582184, 86.8594971, -93.4810638, 79.2129135, -181.8711090, 180.3405457
8: -123.3582306, 84.1924210, -112.3631744, 76.7007446, -200.0589752, 196.5556030
9: -93.5967941, 92.2854080, -85.2954330, 84.0844955, -177.6812897, 177.5808258

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3828749, upper bound: 189.3795593
time: 9.49 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3828749, upper bound: 189.3805831
time: 9.30 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -97.8221436, 77.4975739, -91.7201157, 72.6684341, -170.4905701, 169.2176666
1: -81.4027328, 68.9118500, -76.2232208, 64.5759888, -145.9786835, 145.1350555
2: -107.2190628, 70.6677322, -100.4770584, 66.3441315, -173.5631866, 171.1447754
3: -114.1461868, 61.0313530, -106.9826355, 57.2427826, -171.3889771, 168.0139923
4: -104.3574677, 80.6500320, -97.7633209, 75.5826645, -179.9401245, 178.4133453
5: -94.1122971, 73.5162506, -88.2055893, 68.9242477, -163.0365143, 161.7218323
6: -90.2317123, 86.1987076, -84.6035309, 80.8076172, -171.0392914, 170.8021851
7: -98.0301743, 82.9983749, -91.8867950, 77.8899231, -175.9201050, 174.8851624
8: -117.8125916, 80.4104691, -110.4730682, 75.3866882, -193.1992645, 190.8835144
9: -89.4043274, 88.1451645, -83.8563538, 82.6627655, -172.0670929, 172.0015106

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3828752, upper bound: 189.3795593
time: 7.94 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3828752, upper bound: 189.3806934
time: 8.04 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -92.0358429, 72.9287415, -91.6827011, 72.6037750, -164.6396179, 164.6114502
1: -76.5209274, 64.8197632, -76.2051849, 64.5625610, -141.0834961, 141.0249481
2: -100.8281555, 66.5664673, -100.4302673, 66.3922653, -167.2203979, 166.9967346
3: -107.3605652, 57.4635239, -107.0640106, 57.2560120, -164.6165619, 164.5275116
4: -98.1228561, 75.8779831, -97.6903305, 75.5128403, -173.6356812, 173.5683136
5: -88.5456467, 69.1968536, -88.1595306, 68.8911057, -157.4367523, 157.3563843
6: -84.9043808, 81.0833588, -84.5645676, 80.7399216, -165.6443024, 165.6479187
7: -92.2169952, 78.1573257, -91.9024887, 77.9275894, -170.1445923, 170.0598145
8: -110.8497925, 75.6676331, -110.3649368, 75.2659760, -186.1157684, 186.0325623
9: -84.1467133, 82.9497910, -83.8823166, 82.6267319, -166.7734375, 166.8320618

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3736674, upper bound: 189.3736674
time: 6.43 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3736674, upper bound: 189.3739072
time: 6.36 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -90.6785965, 71.8431549, -87.5010681, 69.2987595, -159.9773407, 159.3441925
1: -75.3463135, 63.8394661, -72.6887741, 61.6127968, -136.9591064, 136.5282288
2: -99.3261490, 65.6059189, -95.8223877, 63.4245796, -162.7507324, 161.4282684
3: -105.7601547, 56.6000938, -102.1568298, 54.6783218, -160.4384766, 158.7569122
4: -96.6350403, 74.7200394, -93.1834488, 72.0647507, -168.6997986, 167.9034882
5: -87.2026825, 68.1417160, -84.1345520, 65.7667847, -152.9694672, 152.2762756
6: -83.6429291, 79.8828201, -80.7137299, 77.0475616, -160.6904755, 160.5965576
7: -90.8396530, 77.0144501, -87.7046814, 74.4264145, -165.2660675, 164.7191162
8: -109.2185440, 74.5288162, -105.3353195, 71.8318787, -181.0504150, 179.8641205
9: -82.9027252, 81.7213669, -80.0789948, 78.8702393, -161.7729645, 161.8003540

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3739072, upper bound: 189.3740715
time: 5.91 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3739072, upper bound: 189.3750171
time: 6.11 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 14.94 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 7, lower bound: -189.3896863, upper bound: 189.3872185
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 7, lower bound: -189.3920735, upper bound: 189.3889863
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 7, lower bound: -189.3896863, upper bound: 189.3872185
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 7, lower bound: -189.3920735, upper bound: 189.3961782
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 7, lower bound: -189.3896863, upper bound: 189.3877364
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 7, lower bound: -189.3920735, upper bound: 189.3895979
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 7, lower bound: -189.3896863, upper bound: 189.3979019
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 7, lower bound: -189.3920735, upper bound: 189.4003660
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 7, lower bound: -189.3828749, upper bound: 189.3795593
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 7, lower bound: -189.3828749, upper bound: 189.3805831
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 7, lower bound: -189.3828752, upper bound: 189.3795593
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 7, lower bound: -189.3828752, upper bound: 189.3806934
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 7, lower bound: -189.3736674, upper bound: 189.3736674
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 7, lower bound: -189.3736674, upper bound: 189.3739072
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 7, lower bound: -189.3739072, upper bound: 189.3740715
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 7, lower bound: -189.3739072, upper bound: 189.3750171

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -90.4456711, 71.6340332, -94.1084747, 74.5433197, -164.9889832, 165.7425079
1: -75.1530685, 63.6199951, -78.2920990, 66.2597733, -141.4128418, 141.9120941
2: -99.0025406, 65.3771133, -103.0751190, 68.0148621, -167.0173950, 168.4522400
3: -105.4132919, 56.4011917, -109.7670212, 58.7031746, -164.1164551, 166.1682129
4: -96.2435379, 74.4630127, -100.2432251, 77.5449753, -173.7885132, 174.7062378
5: -86.9960098, 67.8585587, -90.5216980, 70.6669312, -157.6629333, 158.3802490
6: -83.3775253, 79.5964966, -86.7584686, 82.8547745, -166.2322845, 166.3549652
7: -90.4452744, 76.6751862, -94.1840668, 79.7787933, -170.2240601, 170.8592529
8: -108.9693985, 74.3275757, -113.3543549, 77.3567047, -186.3260651, 187.6819305
9: -82.4697647, 81.3558502, -85.8932877, 84.6926956, -167.1624603, 167.2491302

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 159

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3785232, upper bound: 189.3756480
time: 7.40 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3767571, upper bound: 189.3742579
time: 9.33 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -91.6852417, 72.6138763, -95.1093979, 75.3333817, -167.0186157, 167.7232513
1: -76.1902466, 64.4945984, -79.1181259, 66.9667816, -143.1570282, 143.6127319
2: -100.3727646, 66.2584686, -104.1819153, 68.7282867, -169.1010437, 170.4403839
3: -106.8693161, 57.1694984, -110.9406738, 59.3319511, -166.2012634, 168.1101379
4: -97.5817566, 75.4867401, -101.3399200, 78.3770981, -175.9588318, 176.8266602
5: -88.1910172, 68.7845535, -91.5010300, 71.4272232, -159.6182404, 160.2855530
6: -84.5214920, 80.6980820, -87.6986542, 83.7545853, -168.2760620, 168.3966980
7: -91.6946869, 77.7288208, -95.2045898, 80.6546783, -172.3493195, 172.9334106
8: -110.4609070, 75.3422089, -114.5372314, 78.1721954, -188.6331024, 189.8794403
9: -83.6103363, 82.4798355, -86.8308105, 85.6245117, -169.2348480, 169.3106384

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 159

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3896262, upper bound: 189.3862980
time: 8.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3896262, upper bound: 189.3889862
time: 9.07 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -93.1594238, 73.7881851, -94.1084747, 74.5433197, -167.7027435, 167.8966675
1: -77.4398651, 65.5471649, -78.2920990, 66.2597733, -143.6996460, 143.8392487
2: -102.0113373, 67.3103104, -103.0751190, 68.0148621, -170.0261688, 170.3854370
3: -108.6060410, 58.0801468, -109.7670212, 58.7031746, -167.3092194, 167.8471680
4: -99.2229462, 76.7182846, -100.2432251, 77.5449753, -176.7678986, 176.9615173
5: -89.6149826, 69.9176407, -90.5216980, 70.6669312, -160.2819214, 160.4393311
6: -85.8934479, 82.0109177, -86.7584686, 82.8547745, -168.7481842, 168.7693787
7: -93.2217636, 78.9799652, -94.1840668, 79.7787933, -173.0005341, 173.1640167
8: -112.2351303, 76.5553284, -113.3543549, 77.3567047, -189.5918121, 189.9096832
9: -84.9918976, 83.8191452, -85.8932877, 84.6926956, -169.6846008, 169.7124023

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 159

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3944032, upper bound: 189.3924433
time: 8.80 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3944032, upper bound: 189.3937251
time: 6.94 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -94.3987808, 74.7683334, -95.1093979, 75.3333817, -169.7321625, 169.8777008
1: -78.4771194, 66.4220963, -79.1181259, 66.9667816, -145.4439087, 145.5402222
2: -103.3820114, 68.1917877, -104.1819153, 68.7282867, -172.1102600, 172.3737030
3: -110.0626297, 58.8486404, -110.9406738, 59.3319511, -169.3945770, 169.7893066
4: -100.5612793, 77.7425766, -101.3399200, 78.3770981, -178.9383392, 179.0824890
5: -90.8103638, 70.8442764, -91.5010300, 71.4272232, -162.2375793, 162.3452759
6: -87.0381165, 83.1127014, -87.6986542, 83.7545853, -170.7926788, 170.8113556
7: -94.4716339, 80.0343933, -95.2045898, 80.6546783, -175.1262970, 175.2389832
8: -113.7269058, 77.5702057, -114.5372314, 78.1721954, -191.8991089, 192.1074371
9: -86.1331635, 84.9439545, -86.8308105, 85.6245117, -171.7576752, 171.7747650

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 159

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3954916, upper bound: 189.3932069
time: 8.46 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3954916, upper bound: 189.3961782
time: 8.51 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -90.4456711, 71.6340332, -96.3598633, 76.3310318, -166.7766876, 167.9938965
1: -75.1530685, 63.6199951, -80.1848373, 67.8590088, -143.0120544, 143.8048248
2: -99.0025406, 65.3771133, -105.5722961, 69.6176147, -168.6201477, 170.9494019
3: -105.4132919, 56.4011917, -112.4145660, 60.0955734, -165.5088348, 168.8157654
4: -96.2435379, 74.4630127, -102.7230835, 79.4154892, -175.6590271, 177.1860962
5: -86.9960098, 67.8585587, -92.6936722, 72.3809662, -159.3769836, 160.5522156
6: -83.3775253, 79.5964966, -88.8442001, 84.8550873, -168.2326050, 168.4406586
7: -90.4452744, 76.6751862, -96.4951324, 81.6899033, -172.1351776, 173.1703186
8: -108.9693985, 74.3275757, -116.0555801, 79.2033768, -188.1727448, 190.3831482
9: -82.4697647, 81.3558502, -87.9878464, 86.7371674, -169.2069397, 169.3436890

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 159

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3792294, upper bound: 189.3764275
time: 7.95 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3774766, upper bound: 189.3749736
time: 7.89 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -91.6852417, 72.6138763, -97.5312347, 77.2571869, -168.9424286, 170.1451111
1: -76.1902466, 64.4945984, -81.1566620, 68.6898346, -144.8800812, 145.6512604
2: -100.3727646, 66.2584686, -106.8718719, 70.4522476, -170.8250122, 173.1303406
3: -106.8693161, 57.1694984, -113.7938690, 60.8319397, -167.7012482, 170.9633484
4: -97.5817566, 75.4867401, -104.0047607, 80.3908691, -177.9726105, 179.4915009
5: -88.1910172, 68.7845535, -93.8388138, 73.2714081, -161.4624176, 162.6233368
6: -84.5214920, 80.6980820, -89.9458923, 85.9101181, -170.4316101, 170.6439362
7: -91.6946869, 77.7288208, -97.6903687, 82.7139053, -174.4085846, 175.4191895
8: -110.4609070, 75.3422089, -117.4491425, 80.1604156, -190.6213226, 192.7913513
9: -83.6103363, 82.4798355, -89.0849915, 87.8279343, -171.4382629, 171.5648193

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3901980, upper bound: 189.3869310
time: 7.35 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3901980, upper bound: 189.3895979
time: 7.61 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -93.1594238, 73.7881851, -96.3598633, 76.3310318, -169.4904480, 170.1480408
1: -77.4398651, 65.5471649, -80.1848373, 67.8590088, -145.2988739, 145.7319794
2: -102.0113373, 67.3103104, -105.5722961, 69.6176147, -171.6289062, 172.8825989
3: -108.6060410, 58.0801468, -112.4145660, 60.0955734, -168.7016144, 170.4947205
4: -99.2229462, 76.7182846, -102.7230835, 79.4154892, -178.6384125, 179.4413605
5: -89.6149826, 69.9176407, -92.6936722, 72.3809662, -161.9959412, 162.6112976
6: -85.8934479, 82.0109177, -88.8442001, 84.8550873, -170.7485352, 170.8551178
7: -93.2217636, 78.9799652, -96.4951324, 81.6899033, -174.9116669, 175.4750824
8: -112.2351303, 76.5553284, -116.0555801, 79.2033768, -191.4385071, 192.6109009
9: -84.9918976, 83.8191452, -87.9878464, 86.7371674, -171.7290649, 171.8069763

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 159

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3862367, upper bound: 189.3825558
time: 7.53 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3829954, upper bound: 189.3805305
time: 8.03 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -94.3987808, 74.7683334, -97.5312347, 77.2571869, -171.6559753, 172.2995605
1: -78.4771194, 66.4220963, -81.1566620, 68.6898346, -147.1669464, 147.5787659
2: -103.3820114, 68.1917877, -106.8718719, 70.4522476, -173.8342438, 175.0636444
3: -110.0626297, 58.8486404, -113.7938690, 60.8319397, -170.8945618, 172.6425171
4: -100.5612793, 77.7425766, -104.0047607, 80.3908691, -180.9521027, 181.7473450
5: -90.8103638, 70.8442764, -93.8388138, 73.2714081, -164.0817566, 164.6830597
6: -87.0381165, 83.1127014, -89.9458923, 85.9101181, -172.9482422, 173.0585938
7: -94.4716339, 80.0343933, -97.6903687, 82.7139053, -177.1855469, 177.7247620
8: -113.7269058, 77.5702057, -117.4491425, 80.1604156, -193.8873291, 195.0193481
9: -86.1331635, 84.9439545, -89.0849915, 87.8279343, -173.9610901, 174.0289154

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 159

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3991326, upper bound: 189.3970298
time: 8.11 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3991326, upper bound: 189.4003661
time: 7.48 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -93.0565872, 73.7381821, -93.2918777, 73.9240952, -166.9806671, 167.0300598
1: -77.3798599, 65.5388336, -77.5788803, 65.7078934, -143.0877380, 143.1177063
2: -101.9549713, 67.2846756, -102.2167511, 67.4564590, -169.4114380, 169.5014038
3: -108.5525284, 58.0887947, -108.8348618, 58.2370224, -166.7895508, 166.9236603
4: -99.2262192, 76.7207413, -99.4836807, 76.9181747, -176.1443787, 176.2044220
5: -89.5295334, 69.9597397, -89.7558594, 70.1411133, -159.6706543, 159.7155762
6: -85.8449173, 81.9885712, -86.0633392, 82.1985931, -168.0435181, 168.0519104
7: -93.2390442, 79.0086594, -93.4810638, 79.2129135, -172.4519196, 172.4897003
8: -112.0832596, 76.5089645, -112.3631744, 76.7007446, -188.7839966, 188.8721313
9: -85.0719452, 83.8675766, -85.2954330, 84.0844955, -169.1564331, 169.1629791

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 155

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3788312, upper bound: 189.3755300
time: 8.92 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3817988, upper bound: 189.3782977
time: 8.39 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -91.5829391, 72.5602112, -93.2918777, 73.9240952, -165.5070343, 165.8520813
1: -76.1071320, 64.4763031, -77.5788803, 65.7078934, -141.8150177, 142.0551758
2: -100.3240356, 66.2419968, -102.2167511, 67.4564590, -167.7804871, 168.4587250
3: -106.8158112, 57.1544647, -108.8348618, 58.2370224, -165.0528259, 165.9892883
4: -97.6122513, 75.4664230, -99.4836807, 76.9181747, -174.5304260, 174.9500732
5: -88.0739899, 68.8168716, -89.7558594, 70.1411133, -158.2151031, 158.5727234
6: -84.4759293, 80.6846390, -86.0633392, 82.1985931, -166.6745148, 166.7479858
7: -91.7441788, 77.7682114, -93.4810638, 79.2129135, -170.9570923, 171.2492676
8: -110.3115768, 75.2750473, -112.3631744, 76.7007446, -187.0123291, 187.6382141
9: -83.7222595, 82.5343552, -85.2954330, 84.0844955, -167.8067627, 167.8297729

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3788312, upper bound: 189.3761809
time: 8.19 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3817988, upper bound: 189.3789963
time: 8.50 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -93.0565872, 73.7381821, -91.7201157, 72.6684341, -165.7250061, 165.4582977
1: -77.3798599, 65.5388336, -76.2232208, 64.5759888, -141.9558105, 141.7620544
2: -101.9549713, 67.2846756, -100.4770584, 66.3441315, -168.2991028, 167.7617188
3: -108.5525284, 58.0887947, -106.9826355, 57.2427826, -165.7953033, 165.0714264
4: -99.2262192, 76.7207413, -97.7633209, 75.5826645, -174.8088837, 174.4840698
5: -89.5295334, 69.9597397, -88.2055893, 68.9242477, -158.4537659, 158.1653137
6: -85.8449173, 81.9885712, -84.6035309, 80.8076172, -166.6525269, 166.5920563
7: -93.2390442, 79.0086594, -91.8867950, 77.8899231, -171.1289520, 170.8954163
8: -112.0832596, 76.5089645, -110.4730682, 75.3866882, -187.4699249, 186.9820099
9: -85.0719452, 83.8675766, -83.8563538, 82.6627655, -167.7347107, 167.7239075

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3731454, upper bound: 189.3711927
time: 9.37 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3726595, upper bound: 189.3689105
time: 6.15 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3811233, upper bound: 189.3777757
time: 7.96 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -91.5829391, 72.5602112, -91.7201157, 72.6684341, -164.2513733, 164.2803192
1: -76.1071320, 64.4763031, -76.2232208, 64.5759888, -140.6830902, 140.6995239
2: -100.3240356, 66.2419968, -100.4770584, 66.3441315, -166.6681671, 166.7190552
3: -106.8158112, 57.1544647, -106.9826355, 57.2427826, -164.0585785, 164.1370850
4: -97.6122513, 75.4664230, -97.7633209, 75.5826645, -173.1949158, 173.2297058
5: -88.0739899, 68.8168716, -88.2055893, 68.9242477, -156.9981995, 157.0224609
6: -84.4759293, 80.6846390, -84.6035309, 80.8076172, -165.2835388, 165.2881775
7: -91.7441788, 77.7682114, -91.8867950, 77.8899231, -169.6340942, 169.6549988
8: -110.3115768, 75.2750473, -110.4730682, 75.3866882, -185.6982574, 185.7480927
9: -83.7222595, 82.5343552, -83.8563538, 82.6627655, -166.3850250, 166.3907013

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3782578, upper bound: 189.3763021
time: 6.92 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3811233, upper bound: 189.3789615
time: 9.39 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -92.0358429, 72.9287415, -81.7906876, 64.7897339, -156.8255768, 154.7194214
1: -76.5209274, 64.8197632, -67.8668442, 57.5720291, -134.0929260, 132.6866150
2: -100.8281555, 66.5664673, -89.5122299, 59.3758202, -160.2039795, 156.0786896
3: -107.3605652, 57.4635239, -95.4611053, 51.1614990, -158.5220490, 152.9246216
4: -98.1228561, 75.8779831, -87.0291061, 67.3508453, -165.4736786, 162.9070892
5: -88.5456467, 69.1968536, -78.6395340, 61.4950523, -150.0406952, 147.8363953
6: -84.9043808, 81.0833588, -75.4604340, 71.9997940, -156.9041748, 156.5437927
7: -92.2169952, 78.1573257, -81.9614716, 69.6488342, -161.8658142, 160.1188049
8: -110.8497925, 75.6676331, -98.4710770, 67.1578674, -178.0076599, 174.1387024
9: -84.1467133, 82.9497910, -74.8896561, 73.7427902, -157.8894958, 157.8394165

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 155

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3635870, upper bound: 189.3628100
time: 6.73 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3719092, upper bound: 189.3719094
time: 5.98 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -92.0358429, 72.9287415, -82.2485123, 65.1444626, -157.1802979, 155.1772461
1: -76.5209274, 64.8197632, -68.2226486, 57.8730507, -134.3939819, 133.0424194
2: -100.8281555, 66.5664673, -90.0164413, 59.6965408, -160.5246887, 156.5829010
3: -107.3605652, 57.4635239, -95.9801331, 51.4049530, -158.7655182, 153.4436646
4: -98.1228561, 75.8779831, -87.5041962, 67.6955261, -165.8183441, 163.3821564
5: -88.5456467, 69.1968536, -79.0470428, 61.8092232, -150.3548584, 148.2438965
6: -84.9043808, 81.0833588, -75.8729248, 72.4093323, -157.3137207, 156.9562683
7: -92.2169952, 78.1573257, -82.4151230, 70.0260544, -162.2430420, 160.5724487
8: -110.8497925, 75.6676331, -99.0224686, 67.5038376, -178.3536377, 174.6900940
9: -84.1467133, 82.9497910, -75.2961884, 74.1511993, -158.2979126, 158.2459564

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3635870, upper bound: 189.3630424
time: 6.92 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3719092, upper bound: 189.3721535
time: 5.76 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -90.6785965, 71.8431549, -81.7906876, 64.7897339, -155.4683228, 153.6338501
1: -75.3463135, 63.8394661, -67.8668442, 57.5720291, -132.9183044, 131.7062988
2: -99.3261490, 65.6059189, -89.5122299, 59.3758202, -158.7019653, 155.1181335
3: -105.7601547, 56.6000938, -95.4611053, 51.1614990, -156.9216461, 152.0611725
4: -96.6350403, 74.7200394, -87.0291061, 67.3508453, -163.9858856, 161.7491455
5: -87.2026825, 68.1417160, -78.6395340, 61.4950523, -148.6977386, 146.7812500
6: -83.6429291, 79.8828201, -75.4604340, 71.9997940, -155.6427002, 155.3432617
7: -90.8396530, 77.0144501, -81.9614716, 69.6488342, -160.4884949, 158.9758759
8: -109.2185440, 74.5288162, -98.4710770, 67.1578674, -176.3763885, 172.9998779
9: -82.9027252, 81.7213669, -74.8896561, 73.7427902, -156.6455078, 156.6110077

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 155

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3640392, upper bound: 189.3636062
time: 6.57 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3721535, upper bound: 189.3723329
time: 6.34 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -90.6785965, 71.8431549, -82.2485123, 65.1444626, -155.8230591, 154.0916595
1: -75.3463135, 63.8394661, -68.2226486, 57.8730507, -133.2193604, 132.0620728
2: -99.3261490, 65.6059189, -90.0164413, 59.6965408, -159.0226898, 155.6223450
3: -105.7601547, 56.6000938, -95.9801331, 51.4049530, -157.1651001, 152.5802155
4: -96.6350403, 74.7200394, -87.5041962, 67.6955261, -164.3305664, 162.2242279
5: -87.2026825, 68.1417160, -79.0470428, 61.8092232, -149.0119019, 147.1887512
6: -83.6429291, 79.8828201, -75.8729248, 72.4093323, -156.0522461, 155.7557373
7: -90.8396530, 77.0144501, -82.4151230, 70.0260544, -160.8657074, 159.4295502
8: -109.2185440, 74.5288162, -99.0224686, 67.5038376, -176.7223816, 173.5512543
9: -82.9027252, 81.7213669, -75.2961884, 74.1511993, -157.0539246, 157.0175476

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3640392, upper bound: 189.3642745
time: 5.83 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3721535, upper bound: 189.3733013
time: 5.28 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 11.92 seconds
NS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3785232, upper bound: 189.3756480
NS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3767571, upper bound: 189.3742579
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3896262, upper bound: 189.3862980
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3896262, upper bound: 189.3889862
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3944032, upper bound: 189.3924433
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3944032, upper bound: 189.3937251
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3954916, upper bound: 189.3932069
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3954916, upper bound: 189.3961782
NS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3792294, upper bound: 189.3764275
NS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3774766, upper bound: 189.3749736
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3901980, upper bound: 189.3869310
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3901980, upper bound: 189.3895979
NS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3862367, upper bound: 189.3825558
NS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3829954, upper bound: 189.3805305
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3991326, upper bound: 189.3970298
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3991326, upper bound: 189.4003661
NS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3788312, upper bound: 189.3755300
NS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3817988, upper bound: 189.3782977
NS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3788312, upper bound: 189.3761809
NS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3817988, upper bound: 189.3789963
NS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3726595, upper bound: 189.3689105
NS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3811233, upper bound: 189.3777757
NS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3782578, upper bound: 189.3763021
NS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3811233, upper bound: 189.3789615
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3635870, upper bound: 189.3628100
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3719092, upper bound: 189.3719094
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3635870, upper bound: 189.3630424
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3719092, upper bound: 189.3721535
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3640392, upper bound: 189.3636062
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3721535, upper bound: 189.3723329
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3640392, upper bound: 189.3642745
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.92
Output dim: 7, lower bound: -189.3721535, upper bound: 189.3733013

## BFS NS instance: NS_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -89.1301727, 70.5949097, -86.5503082, 68.5737534, -157.7038727, 157.1452026
1: -74.0562210, 62.6985970, -71.9873810, 60.9639206, -135.0201416, 134.6859436
2: -97.5547485, 64.4536057, -94.7578888, 62.7057800, -160.2604523, 159.2114868
3: -103.8848114, 55.5982208, -100.9797363, 54.0864830, -157.9712677, 156.5779266
4: -94.8295212, 73.3826294, -92.1207733, 71.3371887, -166.1666718, 165.5034027
5: -85.7338943, 66.8805161, -83.2705460, 65.0469131, -150.7807770, 150.1510620
6: -82.1667175, 78.4389877, -79.8006973, 76.2033005, -158.3699951, 158.2396545
7: -89.1292267, 75.5799408, -86.6228485, 73.4832001, -162.6124268, 162.2027893
8: -107.3850708, 73.2429276, -104.2489853, 71.1268387, -178.5119019, 177.4919128
9: -81.2771988, 80.1818008, -79.0389404, 77.9476395, -159.2248383, 159.2207336

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 159

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3767571, upper bound: 189.3742579
time: 6.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3767571, upper bound: 189.3742579
time: 8.12 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -86.8733063, 68.8138504, -86.5876465, 68.6045914, -155.4779053, 155.4014587
1: -72.1705704, 61.1129417, -71.9969177, 60.9569092, -133.1274719, 133.1098633
2: -95.0689621, 62.8668060, -94.7676086, 62.7465210, -157.8154602, 157.6344147
3: -101.2509308, 54.2093544, -101.0026627, 54.0656357, -155.3165588, 155.2119751
4: -92.4001770, 71.5262756, -92.1396561, 71.3446198, -163.7447968, 163.6659241
5: -83.5692825, 65.2046051, -83.2966003, 65.0743637, -148.6436462, 148.5012054
6: -80.0835953, 76.4494629, -79.8002853, 76.2244797, -156.3080750, 156.2497559
7: -86.8727188, 73.7021255, -86.6495132, 73.5339355, -160.4066467, 160.3516235
8: -104.6575241, 71.3800430, -104.2593842, 71.1098328, -175.7673492, 175.6394196
9: -79.2317123, 78.1632538, -79.0961761, 77.9641037, -157.1958160, 157.2594299

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3767571, upper bound: 189.3742580
time: 7.83 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3767571, upper bound: 189.3742580
time: 7.57 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 11.32 + 597.64 = 608.96 seconds
