## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 318.144348814
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519)
1: (-174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401)
2: (-227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989)
3: (-242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914)
4: (-222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943)
5: (-198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390)
6: (-190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228)
7: (-207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249)
8: (-250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678)
9: (-188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716)

## BASE Result
execution time: IAR + LP analysis = 1.23 + 10.83 = 12.07 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -318.2353807, upper bound: 318.2353807


# Binary Search by BASE starts (time budget: 2687.93 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=319.7423400878906
rel_dist={1: [-318.2353263992982, 318.2353263992983]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=319.7423400878906
rel_dist={1: [-318.23529533371016, 318.23529533356407]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=319.7423400878906
rel_dist={1: [-318.2352719030525, 318.23527182273534]}

## Binary Search Result
Binary search time: 42.59 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2645.34 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2273415, upper bound: 318.2263005
time: 8.81 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2235912, upper bound: 318.2235916
time: 7.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.38 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.38
Output dim: 1, lower bound: -318.2273415, upper bound: 318.2263005
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.38
Output dim: 1, lower bound: -318.2235912, upper bound: 318.2235916

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -201.7486572, 159.7551727, -206.9136353, 163.8058472, -365.5545044, 366.6687317
1: -170.0928955, 141.6844482, -174.4282379, 145.3141937, -315.4070435, 316.1126099
2: -222.1750488, 143.9926147, -227.8242950, 147.6721191, -369.8471680, 371.8168945
3: -235.9363251, 124.5760193, -242.0703125, 127.7758865, -363.7121277, 366.6463013
4: -216.7253571, 165.7321167, -222.2664337, 169.9702454, -386.6956177, 387.9985046
5: -193.5281372, 150.6325836, -198.4908142, 154.4979553, -348.0260925, 349.1233826
6: -185.3204346, 178.8934479, -190.0688934, 183.4619141, -368.7823486, 368.9623413
7: -202.6959381, 169.7283020, -207.8973694, 174.0781555, -376.7741089, 377.6256409
8: -243.8268127, 166.6226654, -250.0145874, 170.8148804, -414.6416931, 416.6372681
9: -184.1246033, 181.3619080, -188.8386993, 185.9809875, -370.1055603, 370.2005920

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2235913, upper bound: 318.2235917
time: 7.00 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2235913, upper bound: 318.2235913
time: 7.51 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -204.1518860, 161.7150726, -204.8994446, 162.2227631, -366.3746338, 366.6145020
1: -172.0771484, 143.3549652, -172.7366943, 143.8975525, -315.9747009, 316.0916748
2: -224.8767548, 145.6356659, -225.6212769, 146.2330322, -371.1098022, 371.2569580
3: -238.6658020, 125.9837494, -239.6788483, 126.5283432, -365.1941528, 365.6625671
4: -219.3159637, 167.6087952, -220.0999603, 168.3135681, -387.6295166, 387.7087402
5: -195.8265228, 152.3001862, -196.5534821, 152.9868469, -348.8133545, 348.8536682
6: -187.5478821, 181.0230103, -188.2143097, 181.6791992, -369.2270203, 369.2372742
7: -205.0476685, 171.7090454, -205.8668060, 172.3807831, -377.4284668, 377.5758362
8: -246.7667084, 168.6043701, -247.5997314, 169.1771851, -415.9439087, 416.2041016
9: -186.2518005, 183.5061493, -186.9954529, 184.1760406, -370.4278259, 370.5015869

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2172155, upper bound: 318.2176447
time: 9.01 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2162442, upper bound: 318.2162444
time: 7.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 18.07 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 18.07
Output dim: 1, lower bound: -318.2235913, upper bound: 318.2235917
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 18.07
Output dim: 1, lower bound: -318.2235913, upper bound: 318.2235913
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 18.07
Output dim: 1, lower bound: -318.2172155, upper bound: 318.2176447
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 18.07
Output dim: 1, lower bound: -318.2162442, upper bound: 318.2162444

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -201.7486572, 159.7551727, -201.7486572, 159.7551727, -361.5038147, 361.5038147
1: -170.0928955, 141.6844482, -170.0928955, 141.6844482, -311.7773132, 311.7773132
2: -222.1750488, 143.9926147, -222.1750488, 143.9926147, -366.1676636, 366.1676636
3: -235.9363251, 124.5760193, -235.9363251, 124.5760193, -360.5122375, 360.5122375
4: -216.7253571, 165.7321167, -216.7253571, 165.7321167, -382.4574280, 382.4574280
5: -193.5281372, 150.6325836, -193.5281372, 150.6325836, -344.1607056, 344.1607056
6: -185.3204346, 178.8934479, -185.3204346, 178.8934479, -364.2138367, 364.2138367
7: -202.6959381, 169.7283020, -202.6959381, 169.7283020, -372.4242249, 372.4242249
8: -243.8268127, 166.6226654, -243.8268127, 166.6226654, -410.4494629, 410.4494629
9: -184.1246033, 181.3619080, -184.1246033, 181.3619080, -365.4864807, 365.4864807

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2220316, upper bound: 318.2201471
time: 9.87 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2211406, upper bound: 318.2194761
time: 9.70 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -201.7486572, 159.7551727, -204.1518860, 161.7150726, -363.4637451, 363.9070435
1: -170.0928955, 141.6844482, -172.0771484, 143.3549652, -313.4478760, 313.7615967
2: -222.1750488, 143.9926147, -224.8767548, 145.6356659, -367.8106995, 368.8693848
3: -235.9363251, 124.5760193, -238.6658020, 125.9837494, -361.9199219, 363.2417908
4: -216.7253571, 165.7321167, -219.3159637, 167.6087952, -384.3341370, 385.0480652
5: -193.5281372, 150.6325836, -195.8265228, 152.3001862, -345.8283081, 346.4591064
6: -185.3204346, 178.8934479, -187.5478821, 181.0230103, -366.3433533, 366.4412842
7: -202.6959381, 169.7283020, -205.0476685, 171.7090454, -374.4049377, 374.7759705
8: -243.8268127, 166.6226654, -246.7667084, 168.6043701, -412.4311829, 413.3893738
9: -184.1246033, 181.3619080, -186.2518005, 183.5061493, -367.6306763, 367.6137085

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2220316, upper bound: 318.2201471
time: 8.61 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2211406, upper bound: 318.2194757
time: 9.91 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -204.1518860, 161.7150726, -194.5506897, 154.0735321, -358.2254028, 356.2657471
1: -172.0771484, 143.3549652, -164.0500183, 136.6171722, -308.6942749, 307.4049683
2: -224.8767548, 145.6356659, -214.2755890, 138.8123169, -363.6890869, 359.9112549
3: -238.6658020, 125.9837494, -227.5251007, 120.1183777, -358.7841187, 353.5087585
4: -219.3159637, 167.6087952, -209.0679474, 159.7994080, -379.1153564, 376.6767273
5: -195.8265228, 152.3001862, -186.6442413, 145.2804413, -341.1069641, 338.9444275
6: -187.5478821, 181.0230103, -178.7487183, 172.5274353, -360.0752869, 359.7716675
7: -205.0476685, 171.7090454, -195.4788055, 163.7137451, -368.7613831, 367.1878052
8: -246.7667084, 168.6043701, -235.1565247, 160.6955872, -407.4622803, 403.7608643
9: -186.2518005, 183.5061493, -177.5939636, 174.8778534, -361.1296082, 361.1000977

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2162445, upper bound: 318.2162444
time: 8.04 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2162445, upper bound: 318.2162444
time: 8.39 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -202.4535065, 160.3776093, -197.7071991, 156.5729370, -359.0264282, 358.0847778
1: -170.6532440, 142.1622314, -166.7057343, 138.8411407, -309.4943237, 308.8679810
2: -223.0174103, 144.4195557, -217.7851257, 141.0369873, -364.0543518, 362.2046509
3: -236.6718140, 124.9322510, -231.2283936, 122.0273285, -358.6991272, 356.1606445
4: -217.5031891, 166.2122650, -212.4769897, 162.3713837, -379.8745728, 378.6891479
5: -194.1988525, 151.0358276, -189.6734009, 147.5960693, -341.7948914, 340.7092285
6: -185.9932098, 179.5230713, -181.6598816, 175.3484192, -361.3415527, 361.1829529
7: -203.3458557, 170.2879028, -198.6649017, 166.3729248, -369.7187805, 368.9527588
8: -244.7247467, 167.2119751, -238.9849091, 163.2535706, -407.9783325, 406.1968689
9: -184.7072144, 181.9802094, -180.4600677, 177.7034912, -362.4107056, 362.4402161

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2144047, upper bound: 318.2145773
time: 9.69 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2151028, upper bound: 318.2151028
time: 7.25 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.22 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.22
Output dim: 1, lower bound: -318.2220316, upper bound: 318.2201471
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.22
Output dim: 1, lower bound: -318.2211406, upper bound: 318.2194761
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.22
Output dim: 1, lower bound: -318.2220316, upper bound: 318.2201471
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.22
Output dim: 1, lower bound: -318.2211406, upper bound: 318.2194757
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.22
Output dim: 1, lower bound: -318.2162445, upper bound: 318.2162444
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.22
Output dim: 1, lower bound: -318.2162445, upper bound: 318.2162444
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.22
Output dim: 1, lower bound: -318.2144047, upper bound: 318.2145773
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.22
Output dim: 1, lower bound: -318.2151028, upper bound: 318.2151028

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -191.3622437, 151.5773315, -201.7486572, 159.7551727, -351.1174316, 353.3259277
1: -161.3757477, 134.3794403, -170.0928955, 141.6844482, -303.0601807, 304.4723511
2: -210.7890320, 136.5459595, -222.1750488, 143.9926147, -354.7816467, 358.7210083
3: -223.7425995, 118.1442566, -235.9363251, 124.5760193, -348.3186035, 354.0804749
4: -205.6542664, 157.1881866, -216.7253571, 165.7321167, -371.3863525, 373.9135437
5: -183.5843658, 142.8980103, -193.5281372, 150.6325836, -334.2169495, 336.4261475
6: -175.8215027, 169.7100372, -185.3204346, 178.8934479, -354.7149353, 355.0304260
7: -192.2716980, 161.0302124, -202.6959381, 169.7283020, -361.9999695, 363.7261353
8: -231.3386383, 158.1128540, -243.8268127, 166.6226654, -397.9613037, 401.9396667
9: -174.6902313, 172.0296936, -184.1246033, 181.3619080, -356.0521240, 356.1542053

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2298079, upper bound: 318.2298083
time: 8.61 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2298079, upper bound: 318.2298083
time: 7.92 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -194.6637726, 154.1867218, -200.0654297, 158.4294739, -353.0932617, 354.2521362
1: -164.1523132, 136.7028809, -168.6824799, 140.5024567, -304.6547852, 305.3853760
2: -214.4546814, 138.8712463, -220.3321381, 142.7878265, -357.2424316, 359.2033691
3: -227.6146545, 120.1405029, -233.9605255, 123.5346069, -351.1492615, 354.1009827
4: -209.2142944, 159.8781891, -214.9277496, 164.3482513, -373.5625000, 374.8059387
5: -186.7501068, 145.3206635, -191.9150238, 149.3800201, -336.1300659, 337.2356873
6: -178.8622589, 172.6575470, -183.7791290, 177.4066772, -356.2689209, 356.4366760
7: -195.6023407, 163.8106842, -201.0096741, 168.3207092, -363.9230347, 364.8203430
8: -235.3381195, 160.7822266, -241.8028107, 165.2408752, -400.5789490, 402.5850220
9: -177.6864014, 174.9813385, -182.5938110, 179.8496094, -357.5360107, 357.5751343

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2298079, upper bound: 318.2298079
time: 8.05 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2298079, upper bound: 318.2298079
time: 7.44 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -191.3622437, 151.5773315, -204.1518860, 161.7150726, -353.0773315, 355.7291870
1: -161.3757477, 134.3794403, -172.0771484, 143.3549652, -304.7307129, 306.4566040
2: -210.7890320, 136.5459595, -224.8767548, 145.6356659, -356.4246826, 361.4227295
3: -223.7425995, 118.1442566, -238.6658020, 125.9837494, -349.7262878, 356.8100281
4: -205.6542664, 157.1881866, -219.3159637, 167.6087952, -373.2630615, 376.5041504
5: -183.5843658, 142.8980103, -195.8265228, 152.3001862, -335.8845520, 338.7245483
6: -175.8215027, 169.7100372, -187.5478821, 181.0230103, -356.8444519, 357.2579041
7: -192.2716980, 161.0302124, -205.0476685, 171.7090454, -363.9806824, 366.0778809
8: -231.3386383, 158.1128540, -246.7667084, 168.6043701, -399.9429932, 404.8795776
9: -174.6902313, 172.0296936, -186.2518005, 183.5061493, -358.1963501, 358.2814941

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2211401, upper bound: 318.2194757
time: 9.38 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2211401, upper bound: 318.2194757
time: 9.27 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -194.6637726, 154.1867218, -202.4535065, 160.3776093, -355.0413818, 356.6402283
1: -164.1523132, 136.7028809, -170.6532440, 142.1622314, -306.3145447, 307.3561401
2: -214.4546814, 138.8712463, -223.0174103, 144.4195557, -358.8742065, 361.8885498
3: -227.6146545, 120.1405029, -236.6718140, 124.9322510, -352.5468750, 356.8122864
4: -209.2142944, 159.8781891, -217.5031891, 166.2122650, -375.4264832, 377.3813782
5: -186.7501068, 145.3206635, -194.1988525, 151.0358276, -337.7859497, 339.5194702
6: -178.8622589, 172.6575470, -185.9932098, 179.5230713, -358.3852844, 358.6506958
7: -195.6023407, 163.8106842, -203.3458557, 170.2879028, -365.8902588, 367.1565552
8: -235.3381195, 160.7822266, -244.7247467, 167.2119751, -402.5500488, 405.5069580
9: -177.6864014, 174.9813385, -184.7072144, 181.9802094, -359.6665649, 359.6885376

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2192172, upper bound: 318.2172436
time: 9.18 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2199047, upper bound: 318.2180636
time: 8.63 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -193.9904175, 153.7135620, -194.5506897, 154.0735321, -348.0639648, 348.2642517
1: -163.5468292, 136.2065582, -164.0500183, 136.6171722, -300.1638794, 300.2565918
2: -213.7355347, 138.3475037, -214.2755890, 138.8123169, -352.5477905, 352.6230774
3: -226.7338257, 119.6879959, -227.5251007, 120.1183777, -346.8521729, 347.2130127
4: -208.4832306, 159.2479858, -209.0679474, 159.7994080, -368.2826538, 368.3159180
5: -186.0978394, 144.7321472, -186.6442413, 145.2804413, -331.3782959, 331.3763428
6: -178.2544708, 172.0357971, -178.7487183, 172.5274353, -350.7819214, 350.7844849
7: -194.8470306, 163.1975555, -195.4788055, 163.7137451, -358.5606995, 358.6763611
8: -234.5449371, 160.2731934, -235.1565247, 160.6955872, -395.2404785, 395.4296875
9: -177.0192719, 174.3737640, -177.5939636, 174.8778534, -351.8970947, 351.9677124

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2172155, upper bound: 318.2176447
time: 8.35 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2172159, upper bound: 318.2176451
time: 10.89 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -196.5700226, 155.7570801, -194.5506897, 154.0735321, -350.6435547, 350.3077698
1: -165.7142639, 138.0239410, -164.0500183, 136.6171722, -302.3313599, 302.0739746
2: -216.6136017, 140.1616821, -214.2755890, 138.8123169, -355.4259033, 354.4372559
3: -229.7538605, 121.2414017, -227.5251007, 120.1183777, -349.8722229, 348.7664490
4: -211.2800751, 161.3476715, -209.0679474, 159.7994080, -371.0794678, 370.4156189
5: -188.5694733, 146.6192627, -186.6442413, 145.2804413, -333.8499146, 333.2634583
6: -180.6354980, 174.3458099, -178.7487183, 172.5274353, -353.1629333, 353.0945435
7: -197.4526367, 165.3725281, -195.4788055, 163.7137451, -361.1663513, 360.8512878
8: -237.6834717, 162.3696594, -235.1565247, 160.6955872, -398.3790283, 397.5261536
9: -179.3637695, 176.6831055, -177.5939636, 174.8778534, -354.2415771, 354.2770691

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2172155, upper bound: 318.2176447
time: 8.73 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2172159, upper bound: 318.2176451
time: 8.49 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -196.5642548, 155.6820984, -197.6163483, 156.5004272, -353.0646973, 353.2984619
1: -165.6924438, 138.0314789, -166.6291504, 138.7773895, -304.4698181, 304.6606445
2: -216.4747620, 140.2322388, -217.6841125, 140.9724121, -357.4471741, 357.9163513
3: -229.6332855, 121.2999802, -231.1198273, 121.9712906, -351.6045227, 352.4197998
4: -211.1062775, 161.3355255, -212.3783112, 162.2961273, -373.4024048, 373.7138367
5: -188.5040741, 146.6653900, -189.5854950, 147.5286102, -336.0326843, 336.2508850
6: -180.4830017, 174.2944183, -181.5748444, 175.2677612, -355.7507629, 355.8692322
7: -197.4100647, 165.3611145, -198.5733490, 166.2968903, -363.7069702, 363.9344482
8: -237.5205536, 162.2873840, -238.8737030, 163.1775208, -400.6980591, 401.1610718
9: -179.3304596, 176.6838226, -180.3771667, 177.6218109, -356.9522705, 357.0609436

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2144047, upper bound: 318.2145773
time: 7.51 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2144047, upper bound: 318.2145773
time: 7.90 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -201.0815887, 159.2665100, -197.2728882, 156.2263489, -357.3079224, 356.5393982
1: -169.4696655, 141.1914215, -166.3405457, 138.5366058, -308.0062866, 307.5319214
2: -221.4429016, 143.4110260, -217.3022766, 140.7279663, -362.1708679, 360.7133179
3: -234.9695129, 124.0544739, -230.7098389, 121.7597198, -356.7292480, 354.7643127
4: -215.9790497, 165.0022125, -212.0048523, 162.0113220, -377.9903564, 377.0070190
5: -192.8501740, 150.0152130, -189.2530365, 147.2736969, -340.1238708, 339.2682495
6: -184.6307526, 178.2835541, -181.2525787, 174.9632111, -359.5939636, 359.5361328
7: -201.9304504, 169.1574554, -198.2272339, 166.0101013, -367.9404907, 367.3846436
8: -242.9534760, 165.9669342, -238.4534607, 162.8903503, -405.8437195, 404.4203186
9: -183.4592285, 180.7446442, -180.0642548, 177.3134308, -360.7726440, 360.8088684

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2151028, upper bound: 318.2151029
time: 8.01 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2151028, upper bound: 318.2151029
time: 8.36 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 17.66 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.66
Output dim: 1, lower bound: -318.2298079, upper bound: 318.2298083
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.66
Output dim: 1, lower bound: -318.2298079, upper bound: 318.2298083
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.66
Output dim: 1, lower bound: -318.2298079, upper bound: 318.2298079
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.66
Output dim: 1, lower bound: -318.2298079, upper bound: 318.2298079
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.66
Output dim: 1, lower bound: -318.2211401, upper bound: 318.2194757
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.66
Output dim: 1, lower bound: -318.2211401, upper bound: 318.2194757
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.66
Output dim: 1, lower bound: -318.2192172, upper bound: 318.2172436
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.66
Output dim: 1, lower bound: -318.2199047, upper bound: 318.2180636
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.66
Output dim: 1, lower bound: -318.2172155, upper bound: 318.2176447
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.66
Output dim: 1, lower bound: -318.2172159, upper bound: 318.2176451
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.66
Output dim: 1, lower bound: -318.2172155, upper bound: 318.2176447
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.66
Output dim: 1, lower bound: -318.2172159, upper bound: 318.2176451
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.66
Output dim: 1, lower bound: -318.2144047, upper bound: 318.2145773
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.66
Output dim: 1, lower bound: -318.2144047, upper bound: 318.2145773
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.66
Output dim: 1, lower bound: -318.2151028, upper bound: 318.2151029
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.66
Output dim: 1, lower bound: -318.2151028, upper bound: 318.2151029

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -191.3622437, 151.5773315, -191.3622437, 151.5773315, -342.9395447, 342.9395447
1: -161.3757477, 134.3794403, -161.3757477, 134.3794403, -295.7551880, 295.7551880
2: -210.7890320, 136.5459595, -210.7890320, 136.5459595, -347.3349915, 347.3349915
3: -223.7425995, 118.1442566, -223.7425995, 118.1442566, -341.8868408, 341.8868408
4: -205.6542664, 157.1881866, -205.6542664, 157.1881866, -362.8424683, 362.8424683
5: -183.5843658, 142.8980103, -183.5843658, 142.8980103, -326.4823608, 326.4823608
6: -175.8215027, 169.7100372, -175.8215027, 169.7100372, -345.5315247, 345.5315247
7: -192.2716980, 161.0302124, -192.2716980, 161.0302124, -353.3019104, 353.3019104
8: -231.3386383, 158.1128540, -231.3386383, 158.1128540, -389.4514771, 389.4514771
9: -174.6902313, 172.0296936, -174.6902313, 172.0296936, -346.7199097, 346.7199097

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2285733, upper bound: 318.2281781
time: 8.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2282809, upper bound: 318.2279702
time: 8.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -191.3622437, 151.5773315, -194.6637726, 154.1867218, -345.5489502, 346.2410889
1: -161.3757477, 134.3794403, -164.1523132, 136.7028809, -298.0786133, 298.5317383
2: -210.7890320, 136.5459595, -214.4546814, 138.8712463, -349.6602173, 351.0006409
3: -223.7425995, 118.1442566, -227.6146545, 120.1405029, -343.8830566, 345.7589111
4: -205.6542664, 157.1881866, -209.2142944, 159.8781891, -365.5324707, 366.4024353
5: -183.5843658, 142.8980103, -186.7501068, 145.3206635, -328.9050293, 329.6480713
6: -175.8215027, 169.7100372, -178.8622589, 172.6575470, -348.4790039, 348.5722961
7: -192.2716980, 161.0302124, -195.6023407, 163.8106842, -356.0823364, 356.6325684
8: -231.3386383, 158.1128540, -235.3381195, 160.7822266, -392.1208496, 393.4508972
9: -174.6902313, 172.0296936, -177.6864014, 174.9813385, -349.6715088, 349.7160645

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2285733, upper bound: 318.2281780
time: 7.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2282809, upper bound: 318.2279702
time: 8.96 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -194.6637726, 154.1867218, -191.3622437, 151.5773315, -346.2410889, 345.5489502
1: -164.1523132, 136.7028809, -161.3757477, 134.3794403, -298.5317383, 298.0786133
2: -214.4546814, 138.8712463, -210.7890320, 136.5459595, -351.0006409, 349.6602173
3: -227.6146545, 120.1405029, -223.7425995, 118.1442566, -345.7589111, 343.8830566
4: -209.2142944, 159.8781891, -205.6542664, 157.1881866, -366.4024353, 365.5324707
5: -186.7501068, 145.3206635, -183.5843658, 142.8980103, -329.6480713, 328.9050293
6: -178.8622589, 172.6575470, -175.8215027, 169.7100372, -348.5722961, 348.4790039
7: -195.6023407, 163.8106842, -192.2716980, 161.0302124, -356.6325684, 356.0823364
8: -235.3381195, 160.7822266, -231.3386383, 158.1128540, -393.4508972, 392.1208496
9: -177.6864014, 174.9813385, -174.6902313, 172.0296936, -349.7160645, 349.6715088

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2278708, upper bound: 318.2276518
time: 8.38 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2273525, upper bound: 318.2273524
time: 7.49 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -194.6637726, 154.1867218, -194.6637726, 154.1867218, -348.8504944, 348.8504944
1: -164.1523132, 136.7028809, -164.1523132, 136.7028809, -300.8551941, 300.8551941
2: -214.4546814, 138.8712463, -214.4546814, 138.8712463, -353.3258972, 353.3258972
3: -227.6146545, 120.1405029, -227.6146545, 120.1405029, -347.7551270, 347.7551270
4: -209.2142944, 159.8781891, -209.2142944, 159.8781891, -369.0924377, 369.0924377
5: -186.7501068, 145.3206635, -186.7501068, 145.3206635, -332.0707703, 332.0707703
6: -178.8622589, 172.6575470, -178.8622589, 172.6575470, -351.5197754, 351.5197754
7: -195.6023407, 163.8106842, -195.6023407, 163.8106842, -359.4130249, 359.4130249
8: -235.3381195, 160.7822266, -235.3381195, 160.7822266, -396.1202393, 396.1202393
9: -177.6864014, 174.9813385, -177.6864014, 174.9813385, -352.6676941, 352.6676941

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2278707, upper bound: 318.2276518
time: 7.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2273525, upper bound: 318.2273524
time: 8.89 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -191.3622437, 151.5773315, -193.9904175, 153.7135620, -345.0758057, 345.5677185
1: -161.3757477, 134.3794403, -163.5468292, 136.2065582, -297.5823059, 297.9262695
2: -210.7890320, 136.5459595, -213.7355347, 138.3475037, -349.1364746, 350.2814636
3: -223.7425995, 118.1442566, -226.7338257, 119.6879959, -343.4305725, 344.8780823
4: -205.6542664, 157.1881866, -208.4832306, 159.2479858, -364.9022522, 365.6714172
5: -183.5843658, 142.8980103, -186.0978394, 144.7321472, -328.3165283, 328.9957886
6: -175.8215027, 169.7100372, -178.2544708, 172.0357971, -347.8572693, 347.9645081
7: -192.2716980, 161.0302124, -194.8470306, 163.1975555, -355.4692383, 355.8772583
8: -231.3386383, 158.1128540, -234.5449371, 160.2731934, -391.6118164, 392.6577454
9: -174.6902313, 172.0296936, -177.0192719, 174.3737640, -349.0639954, 349.0489502

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2208067, upper bound: 318.2187273
time: 8.46 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2207325, upper bound: 318.2186846
time: 10.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -191.3622437, 151.5773315, -196.5700226, 155.7570801, -347.1193237, 348.1473389
1: -161.3757477, 134.3794403, -165.7142639, 138.0239410, -299.3996887, 300.0936890
2: -210.7890320, 136.5459595, -216.6136017, 140.1616821, -350.9506836, 353.1595459
3: -223.7425995, 118.1442566, -229.7538605, 121.2414017, -344.9839783, 347.8981323
4: -205.6542664, 157.1881866, -211.2800751, 161.3476715, -367.0019226, 368.4682617
5: -183.5843658, 142.8980103, -188.5694733, 146.6192627, -330.2036133, 331.4674683
6: -175.8215027, 169.7100372, -180.6354980, 174.3458099, -350.1672974, 350.3455200
7: -192.2716980, 161.0302124, -197.4526367, 165.3725281, -357.6441650, 358.4828491
8: -231.3386383, 158.1128540, -237.6834717, 162.3696594, -393.7083130, 395.7962952
9: -174.6902313, 172.0296936, -179.3637695, 176.6831055, -351.3733521, 351.3934631

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2208067, upper bound: 318.2187273
time: 8.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2207325, upper bound: 318.2186846
time: 8.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -194.5751343, 154.1159973, -196.5642548, 155.6820984, -350.2572327, 350.6802368
1: -164.0776062, 136.6407318, -165.6924438, 138.0314789, -302.1090698, 302.3331909
2: -214.3561554, 138.8082275, -216.4747620, 140.2322388, -354.5883789, 355.2829895
3: -227.5088196, 120.0858154, -229.6332855, 121.2999802, -348.8088074, 349.7190857
4: -209.1180267, 159.8047791, -211.1062775, 161.3355255, -370.4535522, 370.9110413
5: -186.6643677, 145.2548828, -188.5040741, 146.6653900, -333.3297424, 333.7589111
6: -178.7792816, 172.5788269, -180.4830017, 174.2944183, -353.0737000, 353.0618286
7: -195.5130463, 163.7365265, -197.4100647, 165.3611145, -360.8741455, 361.1465759
8: -235.2296295, 160.7079926, -237.5205536, 162.2873840, -397.5169983, 398.2285156
9: -177.6055145, 174.9016418, -179.3304596, 176.6838226, -354.2893066, 354.2320862

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2192172, upper bound: 318.2172436
time: 8.52 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2192172, upper bound: 318.2172436
time: 8.42 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -194.2207031, 153.8332367, -201.0815887, 159.2665100, -353.4872131, 354.9148254
1: -163.7796631, 136.3922272, -169.4696655, 141.1914215, -304.9710693, 305.8618774
2: -213.9622650, 138.5559387, -221.4429016, 143.4110260, -357.3732910, 359.9988403
3: -227.0854797, 119.8674545, -234.9695129, 124.0544739, -351.1399536, 354.8369751
4: -208.7325897, 159.5109253, -215.9790497, 165.0022125, -373.7347412, 375.4899902
5: -186.3212891, 144.9917603, -192.8501740, 150.0152130, -336.3364868, 337.8419189
6: -178.4467163, 172.2646484, -184.6307526, 178.2835541, -356.7302856, 356.8953857
7: -195.1558380, 163.4404755, -201.9304504, 169.1574554, -364.3132935, 365.3709106
8: -234.7960052, 160.4118500, -242.9534760, 165.9669342, -400.7629395, 403.3652954
9: -177.2826385, 174.5832977, -183.4592285, 180.7446442, -358.0272217, 358.0425415

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2141019, upper bound: 318.2129500
time: 8.87 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2151029, upper bound: 318.2135942
time: 9.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -193.9904175, 153.7135620, -191.3622437, 151.5773315, -345.5677185, 345.0758057
1: -163.5468292, 136.2065582, -161.3757477, 134.3794403, -297.9262695, 297.5823059
2: -213.7355347, 138.3475037, -210.7890320, 136.5459595, -350.2814636, 349.1364746
3: -226.7338257, 119.6879959, -223.7425995, 118.1442566, -344.8780823, 343.4305725
4: -208.4832306, 159.2479858, -205.6542664, 157.1881866, -365.6714172, 364.9022522
5: -186.0978394, 144.7321472, -183.5843658, 142.8980103, -328.9957886, 328.3165283
6: -178.2544708, 172.0357971, -175.8215027, 169.7100372, -347.9645081, 347.8572693
7: -194.8470306, 163.1975555, -192.2716980, 161.0302124, -355.8772583, 355.4692383
8: -234.5449371, 160.2731934, -231.3386383, 158.1128540, -392.6577454, 391.6118164
9: -177.0192719, 174.3737640, -174.6902313, 172.0296936, -349.0489502, 349.0639954

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2179810, upper bound: 318.2181224
time: 7.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2184289, upper bound: 318.2184289
time: 8.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -193.9904175, 153.7135620, -193.9904175, 153.7135620, -347.7039795, 347.7039795
1: -163.5468292, 136.2065582, -163.5468292, 136.2065582, -299.7533569, 299.7533569
2: -213.7355347, 138.3475037, -213.7355347, 138.3475037, -352.0829468, 352.0829468
3: -226.7338257, 119.6879959, -226.7338257, 119.6879959, -346.4218140, 346.4218140
4: -208.4832306, 159.2479858, -208.4832306, 159.2479858, -367.7312012, 367.7312012
5: -186.0978394, 144.7321472, -186.0978394, 144.7321472, -330.8299561, 330.8299561
6: -178.2544708, 172.0357971, -178.2544708, 172.0357971, -350.2902832, 350.2902832
7: -194.8470306, 163.1975555, -194.8470306, 163.1975555, -358.0445862, 358.0445862
8: -234.5449371, 160.2731934, -234.5449371, 160.2731934, -394.8180847, 394.8180847
9: -177.0192719, 174.3737640, -177.0192719, 174.3737640, -351.3930359, 351.3930359

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2179814, upper bound: 318.2181220
time: 8.98 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2184289, upper bound: 318.2184292
time: 8.76 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -196.5700226, 155.7570801, -191.3622437, 151.5773315, -348.1473389, 347.1193237
1: -165.7142639, 138.0239410, -161.3757477, 134.3794403, -300.0936890, 299.3996887
2: -216.6136017, 140.1616821, -210.7890320, 136.5459595, -353.1595459, 350.9506836
3: -229.7538605, 121.2414017, -223.7425995, 118.1442566, -347.8981323, 344.9839783
4: -211.2800751, 161.3476715, -205.6542664, 157.1881866, -368.4682617, 367.0019226
5: -188.5694733, 146.6192627, -183.5843658, 142.8980103, -331.4674683, 330.2036133
6: -180.6354980, 174.3458099, -175.8215027, 169.7100372, -350.3455200, 350.1672974
7: -197.4526367, 165.3725281, -192.2716980, 161.0302124, -358.4828491, 357.6441650
8: -237.6834717, 162.3696594, -231.3386383, 158.1128540, -395.7962952, 393.7083130
9: -179.3637695, 176.6831055, -174.6902313, 172.0296936, -351.3934631, 351.3733521

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2151858, upper bound: 318.2158841
time: 8.88 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2159403, upper bound: 318.2163549
time: 8.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -196.5700226, 155.7570801, -193.9904175, 153.7135620, -350.2835693, 349.7474976
1: -165.7142639, 138.0239410, -163.5468292, 136.2065582, -301.9208374, 301.5707397
2: -216.6136017, 140.1616821, -213.7355347, 138.3475037, -354.9610901, 353.8971558
3: -229.7538605, 121.2414017, -226.7338257, 119.6879959, -349.4418640, 347.9752197
4: -211.2800751, 161.3476715, -208.4832306, 159.2479858, -370.5280457, 369.8308716
5: -188.5694733, 146.6192627, -186.0978394, 144.7321472, -333.3016357, 332.7170715
6: -180.6354980, 174.3458099, -178.2544708, 172.0357971, -352.6712952, 352.6002808
7: -197.4526367, 165.3725281, -194.8470306, 163.1975555, -360.6502075, 360.2195129
8: -237.6834717, 162.3696594, -234.5449371, 160.2731934, -397.9566650, 396.9145813
9: -179.3637695, 176.6831055, -177.0192719, 174.3737640, -353.7375488, 353.7023926

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 55

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2151858, upper bound: 318.2158841
time: 8.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2159403, upper bound: 318.2163549
time: 8.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -196.5642548, 155.6820984, -194.5751343, 154.1159973, -350.6802368, 350.2572327
1: -165.6924438, 138.0314789, -164.0776062, 136.6407318, -302.3331909, 302.1090698
2: -216.4747620, 140.2322388, -214.3561554, 138.8082275, -355.2829895, 354.5883789
3: -229.6332855, 121.2999802, -227.5088196, 120.0858154, -349.7190857, 348.8088074
4: -211.1062775, 161.3355255, -209.1180267, 159.8047791, -370.9110413, 370.4535522
5: -188.5040741, 146.6653900, -186.6643677, 145.2548828, -333.7589111, 333.3297424
6: -180.4830017, 174.2944183, -178.7792816, 172.5788269, -353.0618286, 353.0737000
7: -197.4100647, 165.3611145, -195.5130463, 163.7365265, -361.1465759, 360.8741455
8: -237.5205536, 162.2873840, -235.2296295, 160.7079926, -398.2285156, 397.5169983
9: -179.3304596, 176.6838226, -177.6055145, 174.9016418, -354.2320862, 354.2893066

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2144047, upper bound: 318.2145773
time: 8.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2144047, upper bound: 318.2145773
time: 8.59 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -196.5642548, 155.6820984, -196.4785614, 155.6841431, -352.2483521, 352.1606445
1: -165.6924438, 138.0314789, -165.6371918, 137.9597626, -303.6522217, 303.6686707
2: -216.4747620, 140.2322388, -216.5120087, 140.0966339, -356.5714111, 356.7442017
3: -229.6332855, 121.2999802, -229.6445618, 121.1849670, -350.8182068, 350.9445496
4: -211.1062775, 161.3355255, -211.1806946, 161.2719421, -372.3781738, 372.5162354
5: -188.5040741, 146.6653900, -188.4810181, 146.5513611, -335.0554199, 335.1464233
6: -180.4830017, 174.2944183, -180.5499268, 174.2645416, -354.7475586, 354.8443604
7: -197.4100647, 165.3611145, -197.3604584, 165.2959900, -362.7060242, 362.7215576
8: -237.5205536, 162.2873840, -237.5715637, 162.2931671, -399.8136902, 399.8589478
9: -179.3304596, 176.6838226, -179.2802582, 176.6008301, -355.9312744, 355.9640198

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2144047, upper bound: 318.2145773
time: 8.41 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2144047, upper bound: 318.2145773
time: 7.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -201.0815887, 159.2665100, -194.2207031, 153.8332367, -354.9148254, 353.4872131
1: -169.4696655, 141.1914215, -163.7796631, 136.3922272, -305.8618774, 304.9710693
2: -221.4429016, 143.4110260, -213.9622650, 138.5559387, -359.9988403, 357.3732910
3: -234.9695129, 124.0544739, -227.0854797, 119.8674545, -354.8369751, 351.1399536
4: -215.9790497, 165.0022125, -208.7325897, 159.5109253, -375.4899902, 373.7347412
5: -192.8501740, 150.0152130, -186.3212891, 144.9917603, -337.8419189, 336.3364868
6: -184.6307526, 178.2835541, -178.4467163, 172.2646484, -356.8953857, 356.7302856
7: -201.9304504, 169.1574554, -195.1558380, 163.4404755, -365.3709106, 364.3132935
8: -242.9534760, 165.9669342, -234.7960052, 160.4118500, -403.3652954, 400.7629395
9: -183.4592285, 180.7446442, -177.2826385, 174.5832977, -358.0425415, 358.0272217

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2151029, upper bound: 318.2151028
time: 8.20 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2151029, upper bound: 318.2151029
time: 8.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -201.0815887, 159.2665100, -196.1584473, 155.4287720, -356.5103455, 355.4249573
1: -169.4696655, 141.1914215, -165.3679199, 137.7353058, -307.2049561, 306.5592957
2: -221.4429016, 143.4110260, -216.1556549, 139.8687439, -361.3116455, 359.5666809
3: -234.9695129, 124.0544739, -229.2624207, 120.9874954, -355.9569702, 353.3168945
4: -215.9790497, 165.0022125, -210.8323669, 161.0063629, -376.9854126, 375.8345032
5: -192.8501740, 150.0152130, -188.1714172, 146.3141479, -339.1643066, 338.1866455
6: -184.6307526, 178.2835541, -180.2492981, 173.9804382, -358.6112061, 358.5328369
7: -201.9304504, 169.1574554, -197.0378265, 165.0287781, -366.9592285, 366.1952515
8: -242.9534760, 165.9669342, -237.1788483, 162.0248260, -404.9782104, 403.1457825
9: -183.4592285, 180.7446442, -178.9886322, 176.3130493, -359.7722778, 359.7332458

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2151028, upper bound: 318.2151029
time: 7.32 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2151028, upper bound: 318.2151028
time: 7.16 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 15.78 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2285733, upper bound: 318.2281781
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2282809, upper bound: 318.2279702
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2285733, upper bound: 318.2281780
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2282809, upper bound: 318.2279702
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2278708, upper bound: 318.2276518
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2273525, upper bound: 318.2273524
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2278707, upper bound: 318.2276518
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2273525, upper bound: 318.2273524
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2208067, upper bound: 318.2187273
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2207325, upper bound: 318.2186846
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2208067, upper bound: 318.2187273
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2207325, upper bound: 318.2186846
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2192172, upper bound: 318.2172436
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2192172, upper bound: 318.2172436
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2141019, upper bound: 318.2129500
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2151029, upper bound: 318.2135942
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2179810, upper bound: 318.2181224
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2184289, upper bound: 318.2184289
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2179814, upper bound: 318.2181220
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2184289, upper bound: 318.2184292
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2151858, upper bound: 318.2158841
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2159403, upper bound: 318.2163549
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2151858, upper bound: 318.2158841
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2159403, upper bound: 318.2163549
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2144047, upper bound: 318.2145773
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2144047, upper bound: 318.2145773
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2144047, upper bound: 318.2145773
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2144047, upper bound: 318.2145773
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2151029, upper bound: 318.2151028
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2151029, upper bound: 318.2151029
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2151028, upper bound: 318.2151029
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 1, lower bound: -318.2151028, upper bound: 318.2151028

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -185.6656647, 147.0319061, -191.2726288, 151.5057983, -337.1714478, 338.3045349
1: -156.5751495, 130.3837280, -161.3001709, 134.3165741, -290.8916931, 291.6838989
2: -204.4553375, 132.4948883, -210.6893616, 136.4822235, -340.9375610, 343.1841736
3: -216.9356995, 114.6305923, -223.6355286, 118.0889664, -335.0246582, 338.2661133
4: -199.4662018, 152.4683228, -205.5569458, 157.1139221, -356.5801392, 358.0252686
5: -178.0734253, 138.6712646, -183.4976501, 142.8314362, -320.9048462, 322.1689148
6: -170.4878845, 164.6504669, -175.7375793, 169.6304169, -340.1182861, 340.3880310
7: -186.5289917, 156.2640991, -192.1813202, 160.9552612, -347.4842529, 348.4453735
8: -224.3605652, 153.3422089, -231.2288818, 158.0377808, -382.3983154, 384.5710754
9: -169.4903717, 166.9067383, -174.6083984, 171.9490967, -341.4394531, 341.5151367

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2294529, upper bound: 318.2294528
time: 8.49 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2294529, upper bound: 318.2294528
time: 7.33 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -189.2967224, 149.9152222, -190.9248505, 151.2282257, -340.5249634, 340.8400269
1: -159.6152954, 132.9242554, -161.0076447, 134.0726471, -293.6878662, 293.9318848
2: -208.4501190, 135.0441437, -210.3026428, 136.2347107, -344.6848145, 345.3468018
3: -221.2141418, 116.8477020, -223.2200470, 117.8745880, -339.0886841, 340.0677185
4: -203.3770447, 155.4152527, -205.1784515, 156.8255768, -360.2026367, 360.5936890
5: -181.5684357, 141.3614349, -183.1608276, 142.5731659, -324.1416016, 324.5222778
6: -173.8169250, 167.8626556, -175.4109497, 169.3219147, -343.1388550, 343.2736206
7: -190.1571503, 159.3164673, -191.8307495, 160.6647644, -350.8218994, 351.1472168
8: -228.7427368, 156.3072510, -230.8034821, 157.7467651, -386.4895020, 387.1107178
9: -172.8089600, 170.1746826, -174.2915192, 171.6365509, -344.4454651, 344.4661560

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 55

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2245163, upper bound: 318.2241439
time: 9.19 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2250295, upper bound: 318.2250292
time: 8.72 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 19.22 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 19.22
Output dim: 1, lower bound: -318.2294529, upper bound: 318.2294528
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 19.22
Output dim: 1, lower bound: -318.2294529, upper bound: 318.2294528
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 19.22
Output dim: 1, lower bound: -318.2245163, upper bound: 318.2241439
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 19.22
Output dim: 1, lower bound: -318.2250295, upper bound: 318.2250292
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2285733, upper bound: 318.2281780
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2282809, upper bound: 318.2279702
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2278708, upper bound: 318.2276518
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2273525, upper bound: 318.2273524
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2278707, upper bound: 318.2276518
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2273525, upper bound: 318.2273524
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2208067, upper bound: 318.2187273
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2207325, upper bound: 318.2186846
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2208067, upper bound: 318.2187273
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2207325, upper bound: 318.2186846
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2192172, upper bound: 318.2172436
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2192172, upper bound: 318.2172436
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2141019, upper bound: 318.2129500
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2151029, upper bound: 318.2135942
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2179810, upper bound: 318.2181224
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2184289, upper bound: 318.2184289
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2179814, upper bound: 318.2181220
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2184289, upper bound: 318.2184292
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2151858, upper bound: 318.2158841
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2159403, upper bound: 318.2163549
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2151858, upper bound: 318.2158841
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2159403, upper bound: 318.2163549
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2144047, upper bound: 318.2145773
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2144047, upper bound: 318.2145773
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2144047, upper bound: 318.2145773
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2144047, upper bound: 318.2145773
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2151029, upper bound: 318.2151028
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2151029, upper bound: 318.2151029
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2151028, upper bound: 318.2151029
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 1, lower bound: -318.2151028, upper bound: 318.2151028
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=319.7423400878906
rel_dist={1: [-318.2353263992982, 318.2353263992983]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2257104, upper bound: 318.2249860
time: 10.07 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2234940, upper bound: 318.2234940
time: 8.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 18.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 18.79
Output dim: 1, lower bound: -318.2257104, upper bound: 318.2249860
IS_A2, status: Status.UNKNOWN, split count: 1, time: 18.79
Output dim: 1, lower bound: -318.2234940, upper bound: 318.2234940

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -201.7486572, 159.7551727, -206.5104065, 163.4895935, -365.2382507, 366.2655640
1: -170.0928955, 141.6844482, -174.0897522, 145.0308533, -315.1237488, 315.7742004
2: -222.1750488, 143.9926147, -227.3832855, 147.3848724, -369.5599060, 371.3759155
3: -235.9363251, 124.5760193, -241.5912323, 127.5261230, -363.4623718, 366.1672363
4: -216.7253571, 165.7321167, -221.8339081, 169.6394043, -386.3647461, 387.5660095
5: -193.5281372, 150.6325836, -198.1033936, 154.1961670, -347.7243042, 348.7359619
6: -185.3204346, 178.8934479, -189.6981659, 183.1052246, -368.4256287, 368.5916138
7: -202.6959381, 169.7283020, -207.4912872, 173.7384796, -376.4343872, 377.2196045
8: -243.8268127, 166.6226654, -249.5316467, 170.4877319, -414.3145142, 416.1542969
9: -184.1246033, 181.3619080, -188.4705811, 185.6203003, -369.7448425, 369.8324890

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2200142, upper bound: 318.2193235
time: 11.73 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2190646, upper bound: 318.2180764
time: 10.84 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -204.1518860, 161.7150726, -201.7696381, 159.7637939, -363.9156799, 363.4847107
1: -172.0771484, 143.3549652, -170.1090393, 141.6964874, -313.7736206, 313.4639893
2: -224.8767548, 145.6356659, -222.1981354, 143.9975586, -368.8743286, 367.8338013
3: -238.6658020, 125.9837494, -235.9638519, 124.5900879, -363.2558594, 361.9475098
4: -219.3159637, 167.6087952, -216.7342072, 165.7396088, -385.0555725, 384.3430176
5: -195.8265228, 152.3001862, -193.5440979, 150.6391907, -346.4656982, 345.8442993
6: -187.5478821, 181.0230103, -185.3328552, 178.9094238, -366.4572754, 366.3558350
7: -205.0476685, 171.7090454, -202.7121735, 169.7430267, -374.7907104, 374.4211731
8: -246.7667084, 168.6043701, -243.8474274, 166.6322632, -413.3989868, 412.4517822
9: -186.2518005, 183.5061493, -184.1313324, 181.3726044, -367.6243896, 367.6374207

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2167858, upper bound: 318.2170999
time: 9.45 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2162169, upper bound: 318.2162169
time: 8.64 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.36 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 19.36
Output dim: 1, lower bound: -318.2200142, upper bound: 318.2193235
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 19.36
Output dim: 1, lower bound: -318.2190646, upper bound: 318.2180764
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 19.36
Output dim: 1, lower bound: -318.2167858, upper bound: 318.2170999
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 19.36
Output dim: 1, lower bound: -318.2162169, upper bound: 318.2162169

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -198.6696930, 157.3305054, -196.1413879, 155.3246002, -353.9942932, 353.4718933
1: -167.5083160, 139.5187073, -165.3863373, 137.7368622, -305.2451782, 304.9050293
2: -218.7995911, 141.7849121, -216.0154114, 139.9501343, -358.7496948, 357.8002930
3: -232.3207397, 122.6694031, -229.4143524, 121.1044159, -353.4251404, 352.0837097
4: -213.4434967, 163.1988068, -210.7802124, 161.1089783, -374.5524902, 373.9790039
5: -190.5802460, 148.3396149, -188.1748199, 146.4749908, -337.0552063, 336.5143738
6: -182.5041199, 176.1707153, -180.2139740, 173.9360352, -356.4401245, 356.3846741
7: -199.6050262, 167.1496735, -197.0832825, 165.0550537, -364.6600952, 364.2329407
8: -240.1240387, 164.0998840, -237.0642242, 161.9902344, -402.1142578, 401.1640625
9: -181.3272705, 178.5946503, -179.0511780, 176.3039856, -357.6312256, 357.6458130

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2186130, upper bound: 318.2177325
time: 10.10 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2187469, upper bound: 318.2178662
time: 10.64 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -196.6762695, 155.7596436, -199.3430481, 157.8583221, -354.5344849, 355.1026611
1: -165.8422241, 138.1225281, -168.0792236, 139.9916534, -305.8338623, 306.2016602
2: -216.6212158, 140.3618317, -219.5731964, 142.2058563, -358.8270264, 359.9350281
3: -229.9823761, 121.4377670, -233.1704712, 123.0404053, -353.0227661, 354.6082153
4: -211.3076935, 161.5612335, -214.2362061, 163.7170105, -375.0246887, 375.7974243
5: -188.6668549, 146.8573608, -191.2466278, 148.8236542, -337.4905090, 338.1040039
6: -180.6754608, 174.4128571, -183.1658630, 176.7959747, -357.4714355, 357.5787354
7: -197.6144562, 165.4860840, -200.3142548, 167.7519073, -365.3662720, 365.8003540
8: -237.7273407, 162.4585571, -240.9453125, 164.5825348, -402.3098755, 403.4038391
9: -179.5115662, 176.8035126, -181.9575195, 179.1691742, -358.6807251, 358.7610474

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2177740, upper bound: 318.2165931
time: 9.41 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2179098, upper bound: 318.2167131
time: 12.69 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -201.1429901, 159.3454285, -191.4629974, 151.6474152, -352.7904053, 350.8084106
1: -169.5512543, 141.2384644, -161.4573212, 134.4451294, -303.9963989, 302.6958008
2: -221.5778503, 143.4774933, -210.8985138, 136.6062775, -358.1841125, 354.3760071
3: -235.1324463, 124.1196365, -223.8589478, 118.2053757, -353.3378296, 347.9785767
4: -216.1080627, 165.1330872, -205.7466431, 157.2597351, -373.3677368, 370.8796692
5: -192.9455566, 150.0597076, -183.6756897, 142.9637299, -335.9092407, 333.7354126
6: -184.7959900, 178.3616180, -175.9062042, 169.7944489, -354.5904541, 354.2677917
7: -202.0270691, 169.1890564, -192.3659973, 161.1108704, -363.1379395, 361.5550537
8: -243.1475067, 166.1369934, -231.4539337, 158.1844177, -401.3319092, 397.5909424
9: -183.5180969, 180.8016815, -174.7677612, 172.1108856, -355.6289673, 355.5694275

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2146441, upper bound: 318.2151969
time: 8.45 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2155608, upper bound: 318.2158747
time: 8.25 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -199.0255890, 157.6777191, -194.5369110, 154.0828400, -353.1083679, 352.2146301
1: -167.7793579, 139.7545929, -164.0425110, 136.6111450, -304.3905029, 303.7970886
2: -219.2645111, 141.9651794, -214.3183746, 138.7739258, -358.0384521, 356.2835083
3: -232.6470184, 122.8097076, -227.4647369, 120.0634232, -352.7104492, 350.2744446
4: -213.8440704, 163.3937683, -209.0682831, 159.7652740, -373.6093445, 372.4620056
5: -190.9129639, 148.4836121, -186.6255951, 145.2186890, -336.1315613, 335.1091309
6: -182.8548431, 176.4953766, -178.7423401, 172.5421753, -355.3970032, 355.2377319
7: -199.9106293, 167.4195404, -195.4684601, 163.7016144, -363.6122437, 362.8880005
8: -240.6031647, 164.4013367, -235.1834106, 160.6785278, -401.2815857, 399.5847473
9: -181.5899811, 178.8998108, -177.5597382, 174.8632965, -356.4532471, 356.4595032

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2142388, upper bound: 318.2143864
time: 8.92 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2150805, upper bound: 318.2150805
time: 8.76 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.96 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.96
Output dim: 1, lower bound: -318.2186130, upper bound: 318.2177325
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.96
Output dim: 1, lower bound: -318.2187469, upper bound: 318.2178662
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.96
Output dim: 1, lower bound: -318.2177740, upper bound: 318.2165931
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.96
Output dim: 1, lower bound: -318.2179098, upper bound: 318.2167131
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.96
Output dim: 1, lower bound: -318.2146441, upper bound: 318.2151969
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.96
Output dim: 1, lower bound: -318.2155608, upper bound: 318.2158747
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.96
Output dim: 1, lower bound: -318.2142388, upper bound: 318.2143864
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.96
Output dim: 1, lower bound: -318.2150805, upper bound: 318.2150805

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -192.9813843, 152.7916107, -193.9918671, 153.6090240, -346.5903931, 346.7834778
1: -162.7150421, 135.5280151, -163.5750275, 136.2287292, -298.9437561, 299.1029968
2: -212.4753876, 137.7390900, -213.6252747, 138.4216156, -350.8970032, 351.3643799
3: -225.5231018, 119.1606445, -226.8453522, 119.7788849, -345.3019104, 346.0059814
4: -207.2640839, 158.4862976, -208.4453888, 159.3277435, -366.5917969, 366.9317017
5: -185.0763855, 144.1188507, -186.0939026, 144.8791809, -329.9555359, 330.2127075
6: -177.1773834, 171.1187286, -178.2013092, 172.0270691, -349.2044373, 349.3200073
7: -193.8713531, 162.3895874, -194.9165955, 163.2560883, -357.1274109, 357.3061523
8: -233.1575623, 159.3356781, -234.4322968, 160.1907501, -393.3483276, 393.7679443
9: -176.1353760, 173.4795074, -177.0888672, 174.3714294, -350.5068054, 350.5683289

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2186130, upper bound: 318.2177325
time: 10.02 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2186130, upper bound: 318.2177325
time: 10.12 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -196.5511017, 155.6256256, -193.9420013, 153.5694122, -350.1205139, 349.5676270
1: -165.7043152, 138.0259705, -163.5350952, 136.1941376, -301.8984070, 301.5610352
2: -216.4022675, 140.2455292, -213.5697632, 138.3847656, -354.7870483, 353.8153076
3: -229.7297668, 121.3406372, -226.7881775, 119.7481155, -349.4778748, 348.1288147
4: -211.1092072, 161.3822327, -208.3878174, 159.2852173, -370.3944092, 369.7700500
5: -188.5131683, 146.7634125, -186.0450439, 144.8418427, -333.3550110, 332.8084412
6: -180.4507904, 174.2769928, -178.1497040, 171.9847870, -352.4355774, 352.4266968
7: -197.4373016, 165.3911133, -194.8663025, 163.2171478, -360.6544495, 360.2574158
8: -237.4640808, 162.2514343, -234.3729553, 160.1490173, -397.6130981, 396.6243896
9: -179.3979950, 176.6922455, -177.0465393, 174.3276825, -353.7256775, 353.7387695

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2187469, upper bound: 318.2178662
time: 10.80 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2187469, upper bound: 318.2178662
time: 10.50 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -191.0308075, 151.2535400, -197.2193146, 156.1634979, -347.1943054, 348.4728394
1: -161.0829468, 134.1611786, -166.2888336, 138.5018005, -299.5846863, 300.4499512
2: -210.3438568, 136.3455353, -217.2122650, 140.6958313, -351.0396729, 353.5578003
3: -223.2363892, 117.9543991, -230.6332550, 121.7301254, -344.9665222, 348.5876465
4: -205.1736145, 156.8829651, -211.9293976, 161.9577026, -367.1313171, 368.8123474
5: -183.2041168, 142.6675568, -189.1909790, 147.2469482, -330.4510498, 331.8585205
6: -175.3880005, 169.3975830, -181.1775360, 174.9099884, -350.2979431, 350.5751343
7: -191.9234772, 160.7613525, -198.1740417, 165.9742584, -357.8977051, 358.9353943
8: -230.8124695, 157.7277679, -238.3461914, 162.8041077, -393.6165466, 396.0739441
9: -174.3573761, 171.7247620, -180.0187988, 177.2592926, -351.6166382, 351.7435608

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2177740, upper bound: 318.2165931
time: 11.18 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2177740, upper bound: 318.2165931
time: 9.78 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -194.5436249, 154.0433655, -197.1168671, 156.0820312, -350.6256104, 351.1601562
1: -164.0261688, 136.6204376, -166.2072144, 138.4304657, -302.4566040, 302.8276062
2: -214.2091980, 138.8115540, -217.0987091, 140.6215210, -354.8306885, 355.9101868
3: -227.3724060, 120.1004944, -230.5119324, 121.6686783, -349.0410461, 350.6124268
4: -208.9590454, 159.7324982, -211.8158722, 161.8713989, -370.8304443, 371.5483704
5: -186.5860291, 145.2709961, -189.0917358, 147.1710510, -333.7570496, 334.3627319
6: -178.6099701, 172.5066071, -181.0779266, 174.8215485, -353.4314270, 353.5845337
7: -195.4324188, 163.7156982, -198.0710602, 165.8916321, -361.3240356, 361.7867432
8: -235.0494537, 160.5993347, -238.2214508, 162.7206879, -397.7701416, 398.8207703
9: -177.5694580, 174.8895264, -179.9285583, 177.1694031, -354.7388000, 354.8180847

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2179098, upper bound: 318.2167134
time: 11.15 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2179098, upper bound: 318.2167134
time: 9.63 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -195.2465210, 154.6443481, -189.3128052, 149.9314880, -345.1780090, 343.9571533
1: -164.5849762, 137.1028748, -159.6453094, 132.9365692, -297.5215149, 296.7481689
2: -215.0272827, 139.2853088, -208.5076294, 135.0768890, -350.1041870, 347.7929382
3: -228.0854034, 120.4826736, -221.2882385, 116.8795471, -344.9649048, 341.7709045
4: -209.7029419, 160.2503815, -203.4100800, 155.4782867, -365.1812134, 363.6604614
5: -187.2439728, 145.6842499, -181.5954437, 141.3683014, -328.6122742, 327.2796936
6: -179.2788086, 173.1267242, -173.8926239, 167.8843842, -347.1631775, 347.0193481
7: -196.0838165, 164.2566681, -190.1981201, 159.3117676, -355.3955688, 354.4547729
8: -235.9346924, 161.2069244, -228.8203125, 156.3834381, -392.3181152, 390.0270996
9: -178.1348724, 175.4987793, -172.8047485, 170.1768341, -348.3117065, 348.3035278

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2086995, upper bound: 318.2089071
time: 9.19 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2094687, upper bound: 318.2101323
time: 7.83 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -199.7855988, 158.2463989, -189.3455048, 149.9584045, -349.7440186, 347.5919189
1: -168.3790436, 140.2769623, -159.6748352, 132.9598846, -301.3389282, 299.9517822
2: -220.0194092, 142.4792175, -208.5426331, 135.0988617, -355.1182861, 351.0218201
3: -233.4473419, 123.2505264, -221.3291626, 116.8998489, -350.3471375, 344.5796814
4: -214.6000061, 163.9349213, -203.4421234, 155.5034943, -370.1034546, 367.3770447
5: -191.6111298, 149.0494843, -181.6270294, 141.3932190, -333.0043030, 330.6765137
6: -183.4466705, 177.1352997, -173.9178772, 167.9146576, -351.3612671, 351.0531616
7: -200.6265869, 168.0701599, -190.2304077, 159.3422241, -359.9687500, 358.3005066
8: -241.3938599, 164.9046478, -228.8591614, 156.4101105, -397.8039246, 393.7637634
9: -182.2832642, 179.5795135, -172.8375397, 170.2069550, -352.4901428, 352.4170532

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2086995, upper bound: 318.2096725
time: 10.31 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2108959, upper bound: 318.2111736
time: 9.68 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -193.1461487, 152.9899750, -192.4086151, 152.3851013, -345.5312195, 345.3985901
1: -162.8260956, 135.6306610, -162.2480469, 135.1181030, -297.9441833, 297.8787231
2: -212.7329712, 137.7849121, -211.9525452, 137.2602539, -349.9932251, 349.7374573
3: -225.6208801, 119.1833801, -224.9208527, 118.7502518, -344.3711243, 344.1041260
4: -207.4572906, 158.5250702, -206.7557983, 158.0025482, -365.4598389, 365.2808838
5: -185.2275848, 144.1201477, -184.5667572, 143.6395264, -328.8671265, 328.6868896
6: -177.3540039, 171.2749481, -176.7493134, 170.6517029, -348.0057068, 348.0242615
7: -193.9850769, 162.5006256, -193.3228760, 161.9209595, -355.9059143, 355.8234863
8: -233.4110565, 159.4842834, -232.5774841, 158.8961029, -392.3071594, 392.0617676
9: -176.2218781, 173.6118774, -175.6168518, 172.9485626, -349.1704407, 349.2286377

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2081977, upper bound: 318.2081236
time: 9.12 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2090955, upper bound: 318.2093983
time: 9.96 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -197.6548157, 156.5679779, -192.3973541, 152.3760223, -350.0308228, 348.9653015
1: -166.5973816, 138.7848206, -162.2429352, 135.1108704, -301.7082520, 301.0277710
2: -217.6913452, 140.9578705, -211.9390869, 137.2507782, -354.9420471, 352.8969727
3: -230.9463043, 121.9327087, -224.9089355, 118.7449570, -349.6912537, 346.8416138
4: -212.3218536, 162.1846161, -206.7406311, 157.9910736, -370.3129272, 368.9252319
5: -189.5656738, 147.4643250, -184.5561218, 143.6320343, -333.1976929, 332.0204468
6: -181.4937134, 175.2571564, -176.7344208, 170.6434174, -352.1371155, 351.9915466
7: -198.4965668, 166.2902985, -193.3115234, 161.9146881, -360.4112549, 359.6017761
8: -238.8339081, 163.1574707, -232.5622864, 158.8869629, -397.7208862, 395.7197571
9: -180.3433075, 177.6660004, -175.6093597, 172.9400635, -353.2833862, 353.2753601

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2091461, upper bound: 318.2087505
time: 9.00 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2105031, upper bound: 318.2105034
time: 8.96 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 19.24 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.24
Output dim: 1, lower bound: -318.2186130, upper bound: 318.2177325
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.24
Output dim: 1, lower bound: -318.2186130, upper bound: 318.2177325
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.24
Output dim: 1, lower bound: -318.2187469, upper bound: 318.2178662
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.24
Output dim: 1, lower bound: -318.2187469, upper bound: 318.2178662
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.24
Output dim: 1, lower bound: -318.2177740, upper bound: 318.2165931
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.24
Output dim: 1, lower bound: -318.2177740, upper bound: 318.2165931
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.24
Output dim: 1, lower bound: -318.2179098, upper bound: 318.2167134
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.24
Output dim: 1, lower bound: -318.2179098, upper bound: 318.2167134
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.24
Output dim: 1, lower bound: -318.2086995, upper bound: 318.2089071
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.24
Output dim: 1, lower bound: -318.2094687, upper bound: 318.2101323
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.24
Output dim: 1, lower bound: -318.2086995, upper bound: 318.2096725
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.24
Output dim: 1, lower bound: -318.2108959, upper bound: 318.2111736
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.24
Output dim: 1, lower bound: -318.2081977, upper bound: 318.2081236
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.24
Output dim: 1, lower bound: -318.2090955, upper bound: 318.2093983
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.24
Output dim: 1, lower bound: -318.2091461, upper bound: 318.2087505
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.24
Output dim: 1, lower bound: -318.2105031, upper bound: 318.2105034

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -192.9813843, 152.7916107, -189.2514648, 149.8925323, -342.8738403, 342.0430908
1: -162.7150421, 135.5280151, -159.5965576, 132.8987274, -295.6137390, 295.1245728
2: -212.4753876, 137.7390900, -208.4418640, 135.0450592, -347.5204468, 346.1809692
3: -225.5231018, 119.1606445, -221.2208099, 116.8421173, -342.3651428, 340.3814697
4: -207.2640839, 158.4862976, -203.3615875, 155.4390717, -362.7030640, 361.8479004
5: -185.0763855, 144.1188507, -181.5412903, 141.3309784, -326.4073486, 325.6601562
6: -177.1773834, 171.1187286, -173.8450317, 167.8354492, -345.0128174, 344.9637451
7: -193.8713531, 162.3895874, -190.1438446, 159.2639771, -353.1353149, 352.5334167
8: -233.1575623, 159.3356781, -228.7531128, 156.3450928, -389.5026550, 388.0887756
9: -176.1353760, 173.4795074, -172.7633667, 170.1316528, -346.2669983, 346.2428589

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2124981, upper bound: 318.2119122
time: 10.14 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2135631, upper bound: 318.2128695
time: 10.45 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -192.9813843, 152.7916107, -191.8247986, 151.9865112, -344.9678955, 344.6163940
1: -162.7150421, 135.5280151, -161.7227631, 134.6876068, -297.4026184, 297.2507935
2: -212.4753876, 137.7390900, -211.3291931, 136.8077545, -349.2831421, 349.0682373
3: -225.5231018, 119.1606445, -224.1451569, 118.3521423, -343.8752136, 343.3057861
4: -207.2640839, 158.4862976, -206.1301575, 157.4544067, -364.7184143, 364.6164551
5: -185.0763855, 144.1188507, -184.0031891, 143.1251831, -328.2015686, 328.1220398
6: -177.1773834, 171.1187286, -176.2277679, 170.1127472, -347.2901306, 347.3464966
7: -193.8713531, 162.3895874, -192.6639862, 161.3858490, -355.2572021, 355.0534973
8: -233.1575623, 159.3356781, -231.8949890, 158.4620209, -391.6195679, 391.2306519
9: -176.1353760, 173.4795074, -175.0420532, 172.4257660, -348.5611572, 348.5215149

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 55

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2124981, upper bound: 318.2119122
time: 9.98 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2135631, upper bound: 318.2128695
time: 11.35 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -196.5511017, 155.6256256, -189.1544037, 149.8156891, -346.3667908, 344.7800293
1: -165.7043152, 138.0259705, -159.5176697, 132.8309326, -298.5352478, 297.5436401
2: -216.4022675, 140.2455292, -208.3339081, 134.9745178, -351.3767700, 348.5794373
3: -229.7297668, 121.3406372, -221.1048737, 116.7831650, -346.5129395, 342.4454956
4: -211.1092072, 161.3822327, -203.2520752, 155.3577576, -366.4669800, 364.6343079
5: -188.5131683, 146.7634125, -181.4473877, 141.2591248, -329.7722778, 328.2107544
6: -180.4507904, 174.2769928, -173.7490082, 167.7508545, -348.2016602, 348.0259705
7: -197.4373016, 165.3911133, -190.0456848, 159.1857910, -356.6231079, 355.4367981
8: -237.4640808, 162.2514343, -228.6364136, 156.2644806, -393.7285461, 390.8878479
9: -179.3979950, 176.6922455, -172.6775818, 170.0449066, -349.4429016, 349.3698120

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2125414, upper bound: 318.2119646
time: 10.10 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2138588, upper bound: 318.2131268
time: 11.17 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -196.5511017, 155.6256256, -191.9133606, 152.0567169, -348.6078186, 347.5390015
1: -165.7043152, 138.0259705, -161.7983246, 134.7492218, -300.4535522, 299.8242798
2: -216.4022675, 140.2455292, -211.4244232, 136.8695526, -353.2718201, 351.6699524
3: -229.7297668, 121.3406372, -224.2539062, 118.4067001, -348.1364136, 345.5945435
4: -211.1092072, 161.3822327, -206.2234039, 157.5253296, -368.6345215, 367.6056213
5: -188.5131683, 146.7634125, -184.0889740, 143.1920319, -331.7052002, 330.8523865
6: -180.4507904, 174.2769928, -176.3048706, 170.1914673, -350.6422729, 350.5817871
7: -197.4373016, 165.3911133, -192.7532501, 161.4631195, -358.9004211, 358.1443176
8: -237.4640808, 162.2514343, -231.9984131, 158.5319977, -395.9960938, 394.2498474
9: -179.3979950, 176.6922455, -175.1260529, 172.5060577, -351.9040527, 351.8182983

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2125414, upper bound: 318.2119646
time: 10.46 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2138588, upper bound: 318.2131268
time: 9.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -191.0308075, 151.2535400, -192.5877380, 152.5298615, -343.5606689, 343.8412781
1: -161.0829468, 134.1611786, -162.4018402, 135.2465515, -296.3294067, 296.5629578
2: -210.3438568, 136.3455353, -212.1464081, 137.3951416, -347.7389832, 348.4919434
3: -223.2363892, 117.9543991, -225.1345520, 118.8594437, -342.0957947, 343.0889587
4: -205.1736145, 156.8829651, -206.9588928, 158.1583862, -363.3320007, 363.8418579
5: -183.2041168, 142.6675568, -184.7411041, 143.7798157, -326.9839172, 327.4086609
6: -175.3880005, 169.3975830, -176.9183655, 170.8132324, -346.2012024, 346.3159180
7: -191.9234772, 160.7613525, -193.5098877, 162.0736084, -353.9970093, 354.2712402
8: -230.8124695, 157.7277679, -232.7962646, 159.0427246, -389.8551636, 390.5240479
9: -174.3573761, 171.7247620, -175.7911072, 173.1139069, -347.4712830, 347.5158691

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2118346, upper bound: 318.2109319
time: 10.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2128036, upper bound: 318.2117715
time: 10.13 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -191.0308075, 151.2535400, -194.4120178, 154.0364532, -345.0672302, 345.6654663
1: -161.0829468, 134.1611786, -163.8959808, 136.5101776, -297.5930786, 298.0570984
2: -210.3438568, 136.3455353, -214.2165985, 138.6277008, -348.9715271, 350.5621338
3: -223.2363892, 117.9543991, -227.1752472, 119.9100800, -343.1463928, 345.1296387
4: -205.1736145, 156.8829651, -208.9358368, 159.5607147, -364.7343140, 365.8187866
5: -183.2041168, 142.6675568, -186.4824677, 145.0177612, -328.2218323, 329.1500244
6: -175.3880005, 169.3975830, -178.6167450, 172.4294739, -347.8174133, 348.0143127
7: -191.9234772, 160.7613525, -195.2778015, 163.5670319, -355.4904175, 356.0391541
8: -230.8124695, 157.7277679, -235.0440674, 160.5653687, -391.3777466, 392.7718506
9: -174.3573761, 171.7247620, -177.3932037, 174.7421722, -349.0995483, 349.1179810

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2118346, upper bound: 318.2109319
time: 9.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2128036, upper bound: 318.2117715
time: 10.22 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -194.5436249, 154.0433655, -192.4351959, 152.4090881, -346.9526978, 346.4785461
1: -164.0261688, 136.6204376, -162.2778473, 135.1401825, -299.1663513, 298.8982849
2: -214.2091980, 138.8115540, -211.9774017, 137.2849121, -351.4940186, 350.7889404
3: -227.3724060, 120.1004944, -224.9520874, 118.7672424, -346.1395874, 345.0525818
4: -208.9590454, 159.7324982, -206.7905884, 158.0308228, -366.9898682, 366.5230713
5: -186.5860291, 145.2709961, -184.5943146, 143.6672363, -330.2532043, 329.8652954
6: -178.6099701, 172.5066071, -176.7716827, 170.6803436, -349.2902832, 349.2782898
7: -195.4324188, 163.7156982, -193.3558960, 161.9488373, -357.3812561, 357.0715942
8: -235.0494537, 160.5993347, -232.6099396, 158.9182892, -393.9677429, 393.2091675
9: -177.5694580, 174.8895264, -175.6550598, 172.9787445, -350.5482178, 350.5445862

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 169

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2118896, upper bound: 318.2110533
time: 10.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2131563, upper bound: 318.2121045
time: 10.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -194.5436249, 154.0433655, -194.4793396, 154.0895233, -348.6331177, 348.5226746
1: -164.0261688, 136.6204376, -163.9552460, 136.5577698, -300.5839233, 300.5756836
2: -214.2091980, 138.8115540, -214.2876587, 138.6738892, -352.8830566, 353.0991821
3: -227.3724060, 120.1004944, -227.2578278, 119.9516602, -347.3239746, 347.3583374
4: -208.9590454, 159.7324982, -209.0060272, 159.6141052, -368.5731506, 368.7385254
5: -186.5860291, 145.2709961, -186.5476074, 145.0697021, -331.6556702, 331.8186035
6: -178.6099701, 172.5066071, -178.6738586, 172.4897919, -351.0996399, 351.1804504
7: -195.4324188, 163.7156982, -195.3456421, 163.6265564, -359.0589600, 359.0613403
8: -235.0494537, 160.5993347, -235.1206055, 160.6178131, -395.6672668, 395.7198792
9: -177.5694580, 174.8895264, -177.4580994, 174.8034821, -352.3728638, 352.3476257

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 169

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2118896, upper bound: 318.2110533
time: 11.23 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2131563, upper bound: 318.2121045
time: 9.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -190.8889771, 151.2154236, -177.6181335, 140.7297821, -331.6187744, 328.8335571
1: -160.9374390, 134.0394135, -149.8565063, 124.7161789, -285.6536255, 283.8959351
2: -210.2552032, 136.1635742, -195.6998444, 126.7018967, -336.9570312, 331.8634033
3: -222.9784546, 117.7712402, -207.5802307, 109.6096802, -332.5881042, 325.3514709
4: -205.0822906, 156.6704865, -191.0083160, 145.8704071, -350.9526978, 347.6787415
5: -183.0697021, 142.4134979, -170.3926239, 132.5932770, -315.6629639, 312.8060608
6: -175.2829285, 169.2980499, -163.1665802, 157.6111603, -332.8941040, 332.4645996
7: -191.7397003, 160.5938263, -178.5369110, 149.4852448, -341.2249451, 339.1307068
8: -230.6958618, 157.6338043, -214.7596436, 146.7983551, -377.4942017, 372.3934326
9: -174.1902008, 171.5956268, -162.2204590, 159.7022247, -333.8924255, 333.8161011

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1830876, upper bound: 318.1811909
time: 10.29 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1757029, upper bound: 318.1762169
time: 9.02 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -191.9884033, 152.0800323, -181.9722290, 144.1608734, -336.1492615, 334.0521545
1: -161.8595581, 134.8122864, -153.5007935, 127.7745361, -289.6340942, 288.3130493
2: -211.4599762, 136.9514465, -200.4742126, 129.8057556, -341.2657166, 337.4255981
3: -224.2604218, 118.4543457, -212.6717987, 112.3045120, -336.5649414, 331.1260986
4: -206.2411652, 157.5732727, -195.6357422, 149.4365540, -355.6777344, 353.2089844
5: -184.1237946, 143.2438354, -174.5691833, 135.8574982, -319.9812927, 317.8130188
6: -176.2848816, 170.2635498, -167.1491241, 161.4360657, -337.7209473, 337.4126587
7: -192.8320770, 161.5170746, -182.8742218, 153.1315002, -345.9635315, 344.3912354
8: -232.0119781, 158.5377808, -219.9816895, 150.3635101, -382.3754578, 378.5194702
9: -175.1812897, 172.5763397, -166.1500244, 163.5866699, -338.7679443, 338.7263794

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1853877, upper bound: 318.1839825
time: 10.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1776349, upper bound: 318.1788239
time: 9.29 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -195.4207458, 154.8117218, -177.6111145, 140.7248077, -336.1455688, 332.4228210
1: -164.7248688, 137.2083588, -149.8527679, 124.7115784, -289.4364319, 287.0610657
2: -215.2388153, 139.3521576, -195.6910858, 126.6955490, -341.9343262, 335.0432129
3: -228.3298492, 120.5339279, -207.5729675, 109.6054688, -337.9353027, 328.1068420
4: -209.9706726, 160.3484497, -190.9971313, 145.8627777, -355.8333740, 351.3455811
5: -187.4299011, 145.7733612, -170.3861542, 132.5886688, -320.0185547, 316.1595154
6: -179.4425354, 173.3004456, -163.1546936, 157.6064758, -337.0490112, 336.4551392
7: -196.2738495, 164.4017029, -178.5290833, 149.4824982, -345.7563477, 342.9307556
8: -236.1456146, 161.3248901, -214.7510834, 146.7919006, -382.9375000, 376.0759888
9: -178.3323364, 175.6698303, -162.2169800, 159.6972046, -338.0295410, 337.8868103

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1857820, upper bound: 318.1835629
time: 11.26 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1805207, upper bound: 318.1802433
time: 9.29 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -196.5511627, 155.7015076, -182.0195923, 144.1991577, -340.7502441, 337.7210999
1: -165.6735382, 138.0031128, -153.5426178, 127.8074799, -293.4809875, 291.5457153
2: -216.4787750, 140.1620941, -200.5255585, 129.8383179, -346.3170776, 340.6876526
3: -229.6521149, 121.2374268, -212.7312622, 112.3341141, -341.9862366, 333.9686584
4: -211.1647797, 161.2772522, -195.6844025, 149.4737549, -360.6385498, 356.9616089
5: -188.5139008, 146.6268768, -174.6147766, 135.8934631, -324.4073486, 321.2416382
6: -180.4756012, 174.2929688, -167.1888275, 161.4786987, -341.9542542, 341.4818115
7: -197.3990631, 165.3508148, -182.9219208, 153.1743622, -350.5734253, 348.2727356
8: -237.4992371, 162.2542267, -220.0380707, 150.4021301, -387.9013367, 382.2922974
9: -179.3511963, 176.6779633, -166.1960144, 163.6302185, -342.9814148, 342.8739624

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1885285, upper bound: 318.1868080
time: 10.36 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1829736, upper bound: 318.1832576
time: 17.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -188.7801666, 149.5542755, -180.6158142, 143.1036224, -331.8837891, 330.1701050
1: -159.1716919, 132.5618744, -152.3739471, 126.8295746, -286.0012817, 284.9357910
2: -207.9522247, 134.6579742, -199.0399323, 128.8134766, -336.7656860, 333.6979065
3: -220.5040436, 116.4665756, -211.0958099, 111.4155502, -331.9195862, 327.5623779
4: -202.8274384, 154.9384460, -194.2466431, 148.3119507, -351.1394043, 349.1850281
5: -181.0452118, 140.8435669, -173.2675476, 134.7900543, -315.8352051, 314.1111145
6: -173.3505554, 167.4392090, -165.9330597, 160.2897644, -333.6402893, 333.3721619
7: -189.6324463, 158.8315735, -181.5634918, 152.0115509, -341.6439819, 340.3950195
8: -228.1623230, 155.9046173, -218.3960266, 149.2260742, -377.3883667, 374.3006592
9: -172.2698975, 169.7013702, -164.9430847, 162.3838654, -334.6537476, 334.6444702

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1820867, upper bound: 318.1798887
time: 11.89 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1752283, upper bound: 318.1754812
time: 10.15 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.34 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.2124981, upper bound: 318.2119122
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.2135631, upper bound: 318.2128695
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.2124981, upper bound: 318.2119122
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.2135631, upper bound: 318.2128695
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.2125414, upper bound: 318.2119646
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.2138588, upper bound: 318.2131268
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.2125414, upper bound: 318.2119646
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.2138588, upper bound: 318.2131268
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.2118346, upper bound: 318.2109319
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.2128036, upper bound: 318.2117715
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.2118346, upper bound: 318.2109319
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.2128036, upper bound: 318.2117715
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.2118896, upper bound: 318.2110533
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.2131563, upper bound: 318.2121045
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.2118896, upper bound: 318.2110533
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.2131563, upper bound: 318.2121045
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.1830876, upper bound: 318.1811909
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.1757029, upper bound: 318.1762169
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.1853877, upper bound: 318.1839825
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.1776349, upper bound: 318.1788239
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.1857820, upper bound: 318.1835629
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.1805207, upper bound: 318.1802433
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.1885285, upper bound: 318.1868080
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.1829736, upper bound: 318.1832576
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.1820867, upper bound: 318.1798887
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 1, lower bound: -318.1752283, upper bound: 318.1754812
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 1, lower bound: -318.2090955, upper bound: 318.2093983
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 1, lower bound: -318.2091461, upper bound: 318.2087505
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 1, lower bound: -318.2105031, upper bound: 318.2105034
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=319.7423400878906
rel_dist={1: [-318.23529533371016, 318.23529533356407]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2242256, upper bound: 318.2239085
time: 13.92 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2233793, upper bound: 318.2233792
time: 11.43 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 25.47 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 25.47
Output dim: 1, lower bound: -318.2242256, upper bound: 318.2239085
IS_A2, status: Status.UNKNOWN, split count: 1, time: 25.47
Output dim: 1, lower bound: -318.2233793, upper bound: 318.2233792

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -201.7486572, 159.7551727, -203.5959167, 161.2037354, -362.9523926, 363.3510742
1: -170.0928955, 141.6844482, -171.6432190, 142.9826508, -313.0755615, 313.3276367
2: -222.1750488, 143.9926147, -224.1955414, 145.3084412, -367.4834595, 368.1881409
3: -235.9363251, 124.5760193, -238.1291046, 125.7204285, -361.6567078, 362.7051086
4: -216.7253571, 165.7321167, -218.7072296, 167.2476349, -383.9729919, 384.4392700
5: -193.5281372, 150.6325836, -195.3028564, 152.0147095, -345.5428467, 345.9354248
6: -185.3204346, 178.8934479, -187.0182953, 180.5269775, -365.8474121, 365.9117432
7: -202.6959381, 169.7283020, -204.5559845, 171.2836761, -373.9796143, 374.2843018
8: -243.8268127, 166.6226654, -246.0400085, 168.1221008, -411.9489136, 412.6626587
9: -184.1246033, 181.3619080, -185.8101349, 183.0137024, -367.1382751, 367.1720581

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2176558, upper bound: 318.2174050
time: 11.90 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2172351, upper bound: 318.2168295
time: 12.79 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -204.1518860, 161.7150726, -197.5976105, 156.4886169, -360.6405029, 359.3126831
1: -172.0771484, 143.3549652, -166.6064758, 138.7621460, -310.8392944, 309.9614258
2: -224.8767548, 145.6356659, -217.6352081, 141.0141907, -365.8909302, 363.2708740
3: -238.6658020, 125.9837494, -231.0102386, 122.0065079, -360.6723022, 356.9939880
4: -219.3159637, 167.6087952, -212.2457123, 162.3077545, -381.6237183, 379.8544922
5: -195.8265228, 152.3001862, -189.5380402, 147.5102081, -343.3367310, 341.8382263
6: -187.5478821, 181.0230103, -181.4918365, 175.2156067, -362.7634277, 362.5148315
7: -205.0476685, 171.7090454, -198.5042572, 166.2260895, -371.2737427, 370.2132263
8: -246.7667084, 168.6043701, -238.8429871, 163.2382507, -410.0049438, 407.4473572
9: -186.2518005, 183.5061493, -180.3117981, 177.6327057, -363.8844910, 363.8179321

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2163640, upper bound: 318.2165155
time: 9.86 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2161527, upper bound: 318.2161527
time: 9.46 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 20.61 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 20.61
Output dim: 1, lower bound: -318.2176558, upper bound: 318.2174050
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 20.61
Output dim: 1, lower bound: -318.2172351, upper bound: 318.2168295
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 20.61
Output dim: 1, lower bound: -318.2163640, upper bound: 318.2165155
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 20.61
Output dim: 1, lower bound: -318.2161527, upper bound: 318.2161527

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -194.2677917, 153.8647461, -193.2182922, 153.0327301, -347.3005371, 347.0830383
1: -163.8136292, 136.4225616, -162.9336090, 135.6831512, -299.4967651, 299.3561707
2: -213.9737854, 138.6288147, -212.8188477, 137.8682098, -351.8419189, 351.4475708
3: -227.1533508, 119.9435425, -225.9449310, 119.2939301, -346.4472656, 345.8883972
4: -208.7510223, 159.5779114, -207.6448822, 158.7109833, -367.4620056, 367.2227783
5: -186.3659973, 145.0615997, -185.3672180, 144.2872009, -330.6531982, 330.4288025
6: -178.4785156, 172.2786713, -177.5272827, 171.3511200, -349.8296509, 349.8058777
7: -195.1871490, 163.4630737, -194.1403656, 162.5933533, -357.7804871, 357.6034546
8: -234.8312073, 160.4932251, -233.5624390, 159.6190338, -394.4502258, 394.0556030
9: -177.3289642, 174.6393738, -176.3837280, 173.6896973, -351.0186768, 351.0230408

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2156248, upper bound: 318.2153299
time: 12.04 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2161979, upper bound: 318.2158848
time: 11.62 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -192.6864014, 152.6163025, -196.4765778, 155.6091766, -348.2955322, 349.0928650
1: -162.4983215, 135.3210297, -165.6736145, 137.9769592, -300.4752502, 300.9946289
2: -212.2523956, 137.5054779, -216.4376221, 140.1631622, -352.4155579, 353.9430847
3: -225.2999115, 118.9689560, -229.7670746, 121.2640839, -346.5639954, 348.7360229
4: -207.0459442, 158.2799530, -211.1599884, 161.3651886, -368.4110718, 369.4398804
5: -184.8434448, 143.8873749, -188.4920044, 146.6778870, -331.5212708, 332.3793945
6: -177.0222931, 170.8896179, -180.5294037, 174.2607422, -351.2830200, 351.4190063
7: -193.6169891, 162.1489716, -197.4280853, 165.3371429, -358.9540405, 359.5770264
8: -232.9290771, 159.1826630, -237.5105896, 162.2545319, -395.1835938, 396.6932373
9: -175.8825989, 173.2172699, -179.3409576, 176.6039124, -352.4864807, 352.5582275

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2152245, upper bound: 318.2148049
time: 14.37 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2158335, upper bound: 318.2154109
time: 10.42 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -196.8378296, 155.9551086, -187.3435822, 148.4141083, -345.2519226, 343.2986755
1: -165.9367065, 138.2094727, -157.9995270, 131.5481720, -297.4848328, 296.2089233
2: -216.8569794, 140.3894501, -206.3936462, 133.6609344, -350.5178833, 346.7830505
3: -230.0769806, 121.4519501, -218.9679718, 115.6548843, -345.7318420, 340.4199219
4: -211.5181732, 161.5904999, -201.3143768, 153.8721008, -365.3902283, 362.9048767
5: -188.8233490, 146.8530273, -179.7210083, 139.8746948, -328.6980591, 326.5740356
6: -180.8582611, 174.5536652, -172.1147766, 166.1473694, -347.0056152, 346.6683655
7: -197.7051392, 165.5823517, -188.2110748, 157.6386871, -355.3438110, 353.7934265
8: -237.9688721, 162.6069336, -226.5118561, 154.8333130, -392.8021851, 389.1187744
9: -179.6058655, 176.9322052, -170.9962463, 168.4185638, -348.0244141, 347.9284668

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2140293, upper bound: 318.2142350
time: 10.58 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2150070, upper bound: 318.2151496
time: 11.89 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -194.9801025, 154.4914398, -190.3277588, 150.7772064, -345.7572937, 344.8192139
1: -164.3878937, 136.9140320, -160.5086060, 133.6506500, -298.0385132, 297.4226379
2: -214.8351593, 139.0690765, -209.7159882, 135.7633972, -350.5985413, 348.7850342
3: -227.8985443, 120.3056107, -222.4684296, 117.4558563, -345.3543701, 342.7740479
4: -209.5250549, 160.0672150, -204.5415955, 156.3043213, -365.8293762, 364.6088257
5: -187.0357361, 145.4714203, -182.5834198, 142.0606079, -329.0963440, 328.0548096
6: -179.1517487, 172.9221191, -174.8676300, 168.8154144, -347.9671631, 347.7897339
7: -195.8565674, 164.0350952, -191.2234802, 160.1536713, -356.0101929, 355.2585754
8: -235.7399292, 161.0825348, -230.1351929, 157.2553864, -392.9953003, 391.2177124
9: -177.9110718, 175.2643127, -173.7064972, 171.0889282, -349.0000000, 348.9707642

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2138630, upper bound: 318.2139456
time: 9.64 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2148280, upper bound: 318.2148280
time: 11.61 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.52 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.52
Output dim: 1, lower bound: -318.2156248, upper bound: 318.2153299
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.52
Output dim: 1, lower bound: -318.2161979, upper bound: 318.2158848
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.52
Output dim: 1, lower bound: -318.2152245, upper bound: 318.2148049
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.52
Output dim: 1, lower bound: -318.2158335, upper bound: 318.2154109
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.52
Output dim: 1, lower bound: -318.2140293, upper bound: 318.2142350
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.52
Output dim: 1, lower bound: -318.2150070, upper bound: 318.2151496
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.52
Output dim: 1, lower bound: -318.2138630, upper bound: 318.2139456
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.52
Output dim: 1, lower bound: -318.2148280, upper bound: 318.2148280

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -188.5752411, 149.3221283, -188.9390259, 149.6174774, -338.1927185, 338.2611694
1: -159.0166473, 132.4295502, -159.3274384, 132.6815033, -291.6981201, 291.7569885
2: -207.6445923, 134.5802612, -208.0610352, 134.8252563, -342.4698486, 342.6412964
3: -220.3500671, 116.4321899, -220.8325195, 116.6544876, -337.0045471, 337.2647095
4: -202.5668945, 154.8615265, -202.9970093, 155.1652985, -357.7321777, 357.8585205
5: -180.8583221, 140.8376160, -181.2255554, 141.1104431, -321.9687500, 322.0631104
6: -173.1481628, 167.2223969, -173.5209503, 167.5510864, -340.6992188, 340.7433167
7: -189.4485779, 158.6996918, -189.8269958, 159.0122223, -348.4607849, 348.5266113
8: -227.8581848, 155.7259674, -228.3221893, 156.0363617, -383.8945312, 384.0481567
9: -172.1325989, 169.5202179, -172.4775848, 169.8424072, -341.9750061, 341.9978027

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2095176, upper bound: 318.2090763
time: 11.09 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2095176, upper bound: 318.2103516
time: 12.10 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -192.1818390, 152.1863403, -189.0711670, 149.7237091, -341.9055481, 341.2575073
1: -162.0370026, 134.9532166, -159.4434357, 132.7744293, -294.8114319, 294.3966675
2: -211.6126862, 137.1125183, -208.2073822, 134.9163208, -346.5289917, 345.3198547
3: -224.6010132, 118.6346359, -220.9913330, 116.7370224, -341.3380432, 339.6259766
4: -206.4519043, 157.7883911, -203.1332550, 155.2722626, -361.7241821, 360.9216309
5: -184.3305817, 143.5101013, -181.3529968, 141.2086029, -325.5391846, 324.8630676
6: -176.4552155, 170.4135132, -173.6346893, 167.6713562, -344.1265564, 344.0481567
7: -193.0523071, 161.7324982, -189.9597321, 159.1280365, -352.1803284, 351.6921387
8: -232.2107391, 158.6709747, -228.4867554, 156.1472931, -388.3580322, 387.1576233
9: -175.4294739, 172.7665100, -172.6034546, 169.9621887, -345.3916626, 345.3699036

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2100559, upper bound: 318.2095643
time: 12.77 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2115770, upper bound: 318.2113075
time: 12.59 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -187.0698853, 148.1332092, -192.2655182, 152.2485046, -339.3183899, 340.3987427
1: -157.7620392, 131.3801422, -162.1234589, 135.0226746, -292.7846985, 293.5036011
2: -206.0069885, 133.5098572, -211.7564545, 137.1685028, -343.1754761, 345.2662659
3: -218.5883942, 115.5032425, -224.7368011, 118.6655807, -337.2539673, 340.2400513
4: -200.9431000, 153.6255493, -206.5856934, 157.8768921, -358.8199768, 360.2112122
5: -179.4084320, 139.7192841, -184.4169312, 143.5514832, -322.9598999, 324.1362000
6: -171.7619934, 165.8993530, -176.5871582, 170.5204163, -342.2823181, 342.4865112
7: -187.9548187, 157.4489288, -193.1841431, 161.8132019, -349.7680054, 350.6329651
8: -226.0492249, 154.4753418, -232.3558807, 158.7273407, -384.7765503, 386.8312378
9: -170.7546692, 168.1640930, -175.4964294, 172.8159943, -343.5706482, 343.6605225

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2091358, upper bound: 318.2085986
time: 10.85 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2102025, upper bound: 318.2098897
time: 12.89 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -190.5561523, 150.9023438, -192.2952423, 152.2736053, -342.8297119, 343.1975708
1: -160.6842041, 133.8204803, -162.1571503, 135.0448914, -295.7290955, 295.9776306
2: -209.8432770, 135.9568024, -211.7899323, 137.1867676, -347.0299988, 347.7467346
3: -222.6923065, 117.6329880, -224.7719574, 118.6871185, -341.3793945, 342.4049377
4: -204.7000427, 156.4530334, -206.6127930, 157.8988190, -362.5988159, 363.0658264
5: -182.7649689, 142.3029633, -184.4463959, 143.5747528, -326.3396912, 326.7493591
6: -174.9589081, 168.9850464, -176.6072845, 170.5513153, -345.5102234, 345.5923157
7: -191.4374847, 160.3808136, -193.2136230, 161.8440857, -353.2815552, 353.5944214
8: -230.2535248, 157.3252106, -232.3922272, 158.7573242, -389.0108337, 389.7174377
9: -173.9428711, 171.3057251, -175.5300293, 172.8466492, -346.7895203, 346.8357239

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2096224, upper bound: 318.2090240
time: 14.04 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2112333, upper bound: 318.2108269
time: 11.67 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -190.9495697, 151.2607117, -182.9736481, 144.9275208, -335.8770752, 334.2342834
1: -160.9784546, 134.0803528, -154.3176117, 128.4827576, -289.4611511, 288.3979187
2: -210.3152771, 136.2039185, -201.5362091, 130.5527954, -340.8680725, 337.7401123
3: -223.0393372, 117.8198547, -213.7444458, 112.9597092, -335.9990540, 331.5642700
4: -205.1219940, 156.7145233, -196.5655975, 150.2523956, -355.3743591, 353.2801208
5: -183.1292114, 142.4842682, -175.4939423, 136.6326599, -319.7618408, 317.9782104
6: -175.3485260, 169.3260345, -168.0227661, 162.2660675, -337.6145935, 337.3488159
7: -191.7709351, 160.6572113, -183.8059692, 153.9826050, -345.7535095, 344.4631958
8: -230.7656708, 157.6837311, -221.1607056, 151.1746368, -381.9402466, 378.8444214
9: -174.2300568, 171.6363220, -167.0068207, 164.4879456, -338.7179871, 338.6431274

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2079728, upper bound: 318.2080409
time: 11.37 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2089566, upper bound: 318.2092199
time: 11.81 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -195.5003357, 154.8720856, -183.4111023, 145.2772217, -340.7774658, 338.2832031
1: -164.7810822, 137.2616119, -154.6896667, 128.7902527, -293.5713196, 291.9512634
2: -215.3206329, 139.4054718, -202.0186768, 130.8617249, -346.1823120, 341.4241028
3: -228.4148407, 120.5949936, -214.2705536, 113.2310715, -341.6458130, 334.8655090
4: -210.0311737, 160.4088440, -197.0345917, 150.6105347, -360.6416626, 357.4433899
5: -187.5080719, 145.8574829, -175.9170990, 136.9583740, -324.4664307, 321.7745667
6: -179.5271606, 173.3448944, -168.4224854, 162.6566315, -342.1837463, 341.7673035
7: -196.3245850, 164.4803619, -184.2456055, 154.3546600, -350.6792603, 348.7259521
8: -236.2392883, 161.3911743, -221.6919250, 151.5382385, -387.7774963, 383.0830994
9: -178.3896027, 175.7281036, -167.4124298, 164.8832245, -343.2727356, 343.1405334

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2088412, upper bound: 318.2087817
time: 10.62 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2104240, upper bound: 318.2105640
time: 14.25 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -189.1210632, 149.8204041, -185.9937134, 147.3203735, -336.4414062, 335.8141174
1: -159.4514923, 132.8046570, -156.8555756, 130.6103668, -290.0618591, 289.6602173
2: -208.3269653, 134.9040375, -204.8991394, 132.6811676, -341.0081177, 339.8031311
3: -220.8971710, 116.6917419, -217.2880707, 114.7822037, -335.6792908, 333.9797974
4: -203.1610107, 155.2155151, -199.8321075, 152.7154388, -355.8763733, 355.0475769
5: -181.3699646, 141.1235504, -178.3916473, 138.8449860, -320.2149658, 319.5151978
6: -173.6701965, 167.7195129, -170.8105774, 164.9656372, -338.6358337, 338.5300903
7: -189.9527130, 159.1334534, -186.8546753, 156.5269775, -346.4796753, 345.9881287
8: -228.5733795, 156.1824188, -224.8290100, 153.6273651, -382.2007446, 381.0114136
9: -172.5616455, 169.9944611, -169.7499390, 167.1904297, -339.7520752, 339.7443542

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2077769, upper bound: 318.2077276
time: 9.78 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2088232, upper bound: 318.2089538
time: 12.71 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -193.6061859, 153.3792572, -186.3638153, 147.6162567, -341.2224426, 339.7430725
1: -163.2031708, 135.9418335, -157.1751099, 130.8716736, -294.0748291, 293.1169434
2: -213.2584381, 138.0596619, -205.3078766, 132.9422913, -346.2007446, 343.3675232
3: -226.1935883, 119.4258957, -217.7359467, 115.0125046, -341.2060547, 337.1618347
4: -207.9997253, 158.8554382, -200.2302399, 153.0183716, -361.0180969, 359.0856934
5: -185.6846008, 144.4503479, -178.7502747, 139.1221161, -324.8067017, 323.2005920
6: -177.7872162, 171.6814728, -171.1483612, 165.2988434, -343.0860596, 342.8298340
7: -194.4401245, 162.9026794, -187.2285919, 156.8446655, -351.2847595, 350.1311951
8: -233.9664612, 159.8362579, -225.2789764, 153.9355316, -387.9019775, 385.1152344
9: -176.6616516, 174.0278168, -170.0951385, 167.5266571, -344.1882935, 344.1229553

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2077769, upper bound: 318.2084121
time: 13.52 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2102708, upper bound: 318.2102704
time: 9.93 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.73 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 1, lower bound: -318.2095176, upper bound: 318.2090763
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 1, lower bound: -318.2095176, upper bound: 318.2103516
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 1, lower bound: -318.2100559, upper bound: 318.2095643
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 1, lower bound: -318.2115770, upper bound: 318.2113075
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 1, lower bound: -318.2091358, upper bound: 318.2085986
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 1, lower bound: -318.2102025, upper bound: 318.2098897
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 1, lower bound: -318.2096224, upper bound: 318.2090240
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 1, lower bound: -318.2112333, upper bound: 318.2108269
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 1, lower bound: -318.2079728, upper bound: 318.2080409
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 1, lower bound: -318.2089566, upper bound: 318.2092199
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 1, lower bound: -318.2088412, upper bound: 318.2087817
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 1, lower bound: -318.2104240, upper bound: 318.2105640
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 1, lower bound: -318.2077769, upper bound: 318.2077276
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 1, lower bound: -318.2088232, upper bound: 318.2089538
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 1, lower bound: -318.2077769, upper bound: 318.2084121
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 1, lower bound: -318.2102708, upper bound: 318.2102704

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -179.9494781, 142.5334167, -177.2713013, 140.4343567, -320.3838501, 319.8047180
1: -151.7942810, 126.3656693, -149.5587921, 124.4795609, -276.2738342, 275.9244690
2: -198.1976776, 128.4021454, -195.2816010, 126.4688110, -324.6664124, 323.6837158
3: -210.2410431, 111.0686493, -207.1564941, 109.4000168, -319.6410522, 318.2251282
4: -193.4199829, 147.7741547, -190.6229401, 145.5782318, -338.9981079, 338.3970642
5: -172.5941162, 134.3658142, -170.0461121, 132.3555756, -304.9496460, 304.4118652
6: -165.2362823, 159.6443939, -162.8188477, 157.3007660, -322.5370483, 322.4632568
7: -180.8472900, 151.4531708, -178.1914825, 149.2090759, -330.0563660, 329.6446533
8: -217.4856567, 148.6545258, -214.2914429, 146.4714050, -363.9570312, 362.9458923
9: -164.3260651, 161.7931213, -161.9172974, 159.3904419, -323.7164917, 323.7104187

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 169

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2085542, upper bound: 318.2081141
time: 11.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2091702, upper bound: 318.2087299
time: 11.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -182.0967255, 144.2250671, -181.6454926, 143.8860016, -325.9826965, 325.8705444
1: -153.5973511, 127.8751221, -153.2194672, 127.5508194, -281.1481323, 281.0945740
2: -200.5517273, 129.9441833, -200.0782776, 129.5877838, -330.1395264, 330.0223694
3: -212.7431946, 112.4001770, -212.2652283, 112.1077576, -324.8509521, 324.6653748
4: -195.6808777, 149.5372620, -195.2711182, 149.1601257, -344.8410034, 344.8083801
5: -174.6556702, 135.9863434, -174.2496948, 135.6394348, -310.2951050, 310.2360229
6: -167.1944733, 161.5279083, -166.8203430, 161.1406250, -328.3350830, 328.3482666
7: -182.9815521, 153.2541504, -182.5469513, 152.8728027, -335.8543701, 335.8010864
8: -220.0576477, 150.4178925, -219.5350952, 150.0524597, -370.1099854, 369.9530029
9: -166.2595520, 163.7100525, -165.8647003, 163.2907562, -329.5502625, 329.5747681

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 169

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2097315, upper bound: 318.2094995
time: 12.28 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2102643, upper bound: 318.2100548
time: 11.85 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -183.5258789, 145.3729401, -177.3291321, 140.4819031, -324.0077820, 322.7019958
1: -154.7883911, 128.8687134, -149.6130676, 124.5209885, -279.3093872, 278.4817810
2: -202.1308289, 130.9126587, -195.3462677, 126.5073776, -328.6381836, 326.2589111
3: -214.4506073, 113.2520370, -207.2248535, 109.4372253, -323.8878174, 320.4768982
4: -197.2699280, 150.6742706, -190.6786194, 145.6237335, -342.8936157, 341.3528748
5: -176.0358276, 137.0152283, -170.1020966, 132.3992310, -308.4350586, 307.1173096
6: -168.5124664, 162.8090668, -162.8634949, 157.3564758, -325.8689270, 325.6725464
7: -184.4187469, 154.4597931, -178.2496185, 149.2627106, -333.6814575, 332.7092896
8: -221.8002319, 151.5736237, -214.3669739, 146.5218658, -368.3220825, 365.9406128
9: -167.5950012, 165.0127106, -161.9756012, 159.4447479, -327.0396729, 326.9882812

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2089665, upper bound: 318.2085030
time: 12.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2096774, upper bound: 318.2092114
time: 13.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -185.7378845, 147.1161804, -181.7224426, 143.9460449, -329.6838684, 328.8386230
1: -156.6453400, 130.4228973, -153.2894135, 127.6052551, -284.2505188, 283.7123108
2: -204.5582886, 132.5003052, -200.1649628, 129.6404877, -334.1987915, 332.6652832
3: -217.0377960, 114.6250229, -212.3660278, 112.1557465, -329.1935120, 326.9910583
4: -199.6052856, 152.4913635, -195.3513184, 149.2221069, -348.8273621, 347.8426819
5: -178.1604462, 138.6844177, -174.3186951, 135.6923828, -313.8527832, 313.0031128
6: -170.5349426, 164.7492371, -166.8855591, 161.2143402, -331.7492676, 331.6347961
7: -186.6208344, 156.3162537, -182.6276855, 152.9416962, -339.5624695, 338.9438782
8: -224.4503784, 153.3912659, -219.6368408, 150.1202393, -374.5706177, 373.0280762
9: -169.5880737, 166.9862823, -165.9413910, 163.3638153, -332.9519043, 332.9276733

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 68

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2106812, upper bound: 318.2104111
time: 13.31 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2111308, upper bound: 318.2108352
time: 11.46 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -178.3658600, 141.2826843, -180.4884338, 142.9789124, -321.3447571, 321.7711182
1: -150.4739838, 125.2613907, -152.2615051, 126.7442780, -277.2182617, 277.5228882
2: -196.4760742, 127.2765732, -198.8608856, 128.7337036, -325.2097473, 326.1373596
3: -208.3860168, 110.0903320, -210.9298096, 111.3402176, -319.7262268, 321.0200806
4: -191.7107849, 146.4734955, -194.0929413, 148.1983643, -339.9091187, 340.5664368
5: -171.0689240, 133.1885834, -173.1315308, 134.7142792, -305.7832031, 306.3200684
6: -163.7786560, 158.2522278, -165.7844543, 160.1726074, -323.9512024, 324.0366821
7: -179.2756805, 150.1356964, -181.4396973, 151.9173889, -331.1930542, 331.5753784
8: -215.5814056, 147.3396606, -218.1908264, 149.0710297, -364.6524353, 365.5304871
9: -162.8769073, 160.3668671, -164.8372345, 162.2650299, -325.1419373, 325.2041016

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 169

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2082329, upper bound: 318.2076829
time: 12.50 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2087831, upper bound: 318.2082192
time: 11.71 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -180.6400757, 143.0742340, -184.9201660, 146.4737396, -327.1137695, 327.9943542
1: -152.3845673, 126.8602219, -155.9773407, 129.8580933, -282.2426147, 282.8375549
2: -198.9664917, 128.9087677, -203.7168732, 131.8976135, -330.8641052, 332.6255188
3: -211.0394440, 111.5030746, -216.1134796, 114.0875397, -325.1269836, 327.6165466
4: -194.1118622, 148.3417664, -198.8098907, 151.8310089, -345.9428711, 347.1516418
5: -173.2532349, 134.9033356, -177.3864746, 138.0382080, -311.2914429, 312.2897339
6: -165.8538513, 160.2488098, -169.8404694, 164.0696869, -329.9235229, 330.0892334
7: -181.5367737, 152.0441895, -185.8570099, 155.6307983, -337.1675415, 337.9011841
8: -218.3088074, 149.2090607, -223.5138855, 152.7069092, -371.0157166, 372.7228699
9: -164.9266968, 162.3968353, -168.8402863, 166.2211304, -331.1478271, 331.2370911

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 169

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2094372, upper bound: 318.2091142
time: 12.32 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2099066, upper bound: 318.2095608
time: 12.22 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -181.8180542, 144.0246124, -180.4408875, 142.9432373, -324.7612610, 324.4655151
1: -153.3659210, 127.6784134, -152.2293396, 126.7127228, -280.0786438, 279.9077454
2: -200.2733002, 129.6987305, -198.8087006, 128.6979065, -328.9711914, 328.5074463
3: -212.4464874, 112.1980515, -210.8714752, 111.3135910, -323.7600708, 323.0695190
4: -195.4293823, 149.2706299, -194.0355835, 148.1562805, -343.5856628, 343.3062134
5: -174.3912811, 135.7469482, -173.0863647, 134.6802826, -309.0714722, 308.8333130
6: -166.9410248, 161.3074951, -165.7318115, 160.1354675, -327.0764771, 327.0392761
7: -182.7219238, 153.0396423, -181.3903198, 151.8844604, -334.6063538, 334.4299622
8: -219.7432098, 150.1594238, -218.1340637, 149.0363770, -368.7795410, 368.2934875
9: -166.0338440, 163.4779358, -164.8002930, 162.2265930, -328.2604065, 328.2782288

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2085855, upper bound: 318.2080281
time: 11.53 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2092469, upper bound: 318.2086741
time: 13.63 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -184.1657410, 145.8745270, -184.9960785, 146.5346222, -330.7003174, 330.8706055
1: -155.3388214, 129.3282623, -156.0478821, 129.9118500, -285.2506714, 285.3761597
2: -202.8456268, 131.3831635, -203.8004303, 131.9476471, -334.7932739, 335.1835938
3: -215.1925964, 113.6573334, -216.2052765, 114.1367416, -329.3293457, 329.8625793
4: -197.9121246, 151.2003326, -198.8860779, 151.8905029, -349.8026123, 350.0864258
5: -176.6463623, 137.5172119, -177.4590607, 138.0959320, -314.7422485, 314.9762573
6: -169.0882568, 163.3691101, -169.9033661, 164.1403961, -333.2285767, 333.2724304
7: -185.0586700, 155.0093994, -185.9320374, 155.6995087, -340.7581787, 340.9414062
8: -222.5590363, 152.0900116, -223.6051941, 152.7714844, -375.3305054, 375.6951904
9: -168.1501007, 165.5733185, -168.9138641, 166.2921600, -334.4422607, 334.4870911

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2094372, upper bound: 318.2099458
time: 14.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2107710, upper bound: 318.2103635
time: 11.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -182.3817444, 144.5197601, -171.3252869, 135.7627258, -318.1444702, 315.8450317
1: -153.8072968, 128.0569153, -144.5688934, 120.2942734, -274.1015015, 272.6257935
2: -200.9315948, 130.0662537, -188.7782745, 122.2097931, -323.1413269, 318.8445435
3: -212.9976654, 112.4889603, -200.0918884, 105.7162781, -318.7139282, 312.5808105
4: -196.0379333, 149.6758118, -184.2142639, 140.6837463, -336.7216797, 333.8900452
5: -174.9217529, 136.0537262, -164.3361969, 127.8920288, -302.8137207, 300.3899231
6: -167.4918060, 161.7979431, -157.3395081, 152.0328369, -319.5246277, 319.1374512
7: -183.2294159, 153.4549713, -172.1920624, 144.1927032, -327.4221191, 325.6469727
8: -220.4640503, 150.6593475, -207.1562042, 141.6268158, -362.0908508, 357.8155212
9: -166.4742584, 163.9610748, -156.4631195, 154.0537872, -320.5280457, 320.4241638

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1772503, upper bound: 318.1764618
time: 12.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1745167, upper bound: 318.1746784
time: 13.02 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 26.89 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.89
Output dim: 1, lower bound: -318.2085542, upper bound: 318.2081141
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.89
Output dim: 1, lower bound: -318.2091702, upper bound: 318.2087299
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.89
Output dim: 1, lower bound: -318.2097315, upper bound: 318.2094995
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.89
Output dim: 1, lower bound: -318.2102643, upper bound: 318.2100548
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.89
Output dim: 1, lower bound: -318.2089665, upper bound: 318.2085030
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.89
Output dim: 1, lower bound: -318.2096774, upper bound: 318.2092114
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.89
Output dim: 1, lower bound: -318.2106812, upper bound: 318.2104111
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.89
Output dim: 1, lower bound: -318.2111308, upper bound: 318.2108352
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.89
Output dim: 1, lower bound: -318.2082329, upper bound: 318.2076829
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.89
Output dim: 1, lower bound: -318.2087831, upper bound: 318.2082192
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.89
Output dim: 1, lower bound: -318.2094372, upper bound: 318.2091142
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.89
Output dim: 1, lower bound: -318.2099066, upper bound: 318.2095608
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.89
Output dim: 1, lower bound: -318.2085855, upper bound: 318.2080281
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.89
Output dim: 1, lower bound: -318.2092469, upper bound: 318.2086741
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.89
Output dim: 1, lower bound: -318.2094372, upper bound: 318.2099458
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.89
Output dim: 1, lower bound: -318.2107710, upper bound: 318.2103635
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.89
Output dim: 1, lower bound: -318.1772503, upper bound: 318.1764618
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.89
Output dim: 1, lower bound: -318.1745167, upper bound: 318.1746784
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 1, lower bound: -318.2089566, upper bound: 318.2092199
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 1, lower bound: -318.2088412, upper bound: 318.2087817
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 1, lower bound: -318.2104240, upper bound: 318.2105640
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 1, lower bound: -318.2077769, upper bound: 318.2077276
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 1, lower bound: -318.2088232, upper bound: 318.2089538
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 1, lower bound: -318.2077769, upper bound: 318.2084121
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 1, lower bound: -318.2102708, upper bound: 318.2102704
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=319.7423400878906
rel_dist={1: [-318.2352719030525, 318.23527182273534]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1855.69 seconds
