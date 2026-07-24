## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 173.89956106530002
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329)
1: (-79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183)
2: (-104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471)
3: (-110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219)
4: (-101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509)
5: (-90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867)
6: (-86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223)
7: (-95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773)
8: (-114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580)
9: (-86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454)

## BASE Result
execution time: IAR + LP analysis = 1.24 + 9.57 = 10.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -174.0741399, upper bound: 174.0741399


# Binary Search by BASE starts (time budget: 1989.19 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=175.32177734375
rel_dist={7: [-174.07406456819004, 174.07406456764704]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=175.32177734375
rel_dist={7: [-174.07363473064066, 174.07363473064066]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=175.32177734375
rel_dist={7: [-174.07321147769613, 174.07321147870266]}

## Binary Search Result
Binary search time: 39.02 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1950.17 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047457, upper bound: 174.0136738
time: 7.21 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9872560, upper bound: 173.9872560
time: 6.51 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.86 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.86
Output dim: 7, lower bound: -174.0047457, upper bound: 174.0136738
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.86
Output dim: 7, lower bound: -173.9872560, upper bound: 173.9872560

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -94.2460861, 74.8623047, -94.5060349, 75.0702057, -169.3162842, 169.3683319
1: -78.9821320, 66.4131470, -79.2014389, 66.5985794, -145.5807037, 145.6145935
2: -104.0140457, 67.8887329, -104.3030472, 68.0764999, -172.0905304, 172.1917725
3: -110.3552094, 58.0360298, -110.6649246, 58.1981163, -168.5533295, 168.7009277
4: -100.8139877, 77.5684891, -101.0963440, 77.7846146, -178.5986023, 178.6648254
5: -90.4393463, 70.3454666, -90.6905060, 70.5433807, -160.9827271, 161.0359802
6: -86.6970139, 83.5235748, -86.9384842, 83.7556839, -170.4526825, 170.4620209
7: -94.8713150, 79.9647064, -95.1351624, 80.1866226, -175.0579376, 175.0998688
8: -114.1304474, 77.2884140, -114.4460297, 77.5040588, -191.6344910, 191.7344360
9: -86.4736252, 84.5179214, -86.7146835, 84.7555695, -171.2291870, 171.2326050

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9803800, upper bound: 173.9852296
time: 9.95 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9891405, upper bound: 173.9935786
time: 9.16 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9519517, upper bound: 173.9511828
time: 7.56 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9694898, upper bound: 173.9700041
time: 8.67 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0038496, upper bound: 174.0126134
time: 8.26 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9961766, upper bound: 174.0059599
time: 7.99 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -97.1398468, 77.0113602, -94.1327133, 74.7713242, -171.9111633, 171.1440735
1: -81.2423706, 68.2438812, -78.8861313, 66.3316269, -147.5739899, 147.1299744
2: -106.9815216, 69.6908875, -103.8874664, 67.8059921, -174.7875061, 173.5783386
3: -113.4796600, 59.5425911, -110.2196121, 57.9647293, -171.4443970, 169.7622070
4: -103.7134476, 79.6397400, -100.6903534, 77.4732971, -181.1867371, 180.3300629
5: -93.0929413, 72.0213623, -90.3295212, 70.2579880, -163.3509216, 162.3508759
6: -89.2468872, 85.9368439, -86.5912552, 83.4216995, -172.6685791, 172.5280762
7: -97.5071945, 82.1068726, -94.7553101, 79.8670197, -177.3742065, 176.8621826
8: -117.5732346, 79.3855209, -113.9926224, 77.1937714, -194.7669983, 193.3781281
9: -88.7992706, 86.7623596, -86.3675461, 84.4130859, -173.2123566, 173.1298828

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9804029, upper bound: 173.9789380
time: 6.43 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9765027, upper bound: 173.9765026
time: 6.51 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 43.97 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 43.97
Output dim: 7, lower bound: -174.0038496, upper bound: 174.0126134
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 43.97
Output dim: 7, lower bound: -173.9961766, upper bound: 174.0059599
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 43.97
Output dim: 7, lower bound: -173.9804029, upper bound: 173.9789380
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 43.97
Output dim: 7, lower bound: -173.9765027, upper bound: 173.9765026

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -94.2460861, 74.8623047, -91.0917587, 72.3756943, -166.6217804, 165.9540253
1: -78.9821320, 66.4131470, -76.3301773, 64.1884613, -143.1705780, 142.7433167
2: -104.0140457, 67.8887329, -100.5319138, 65.6316681, -169.6456909, 168.4206543
3: -110.3552094, 58.0360298, -106.6444931, 56.0903130, -166.4455261, 164.6805267
4: -100.8139877, 77.5684891, -97.4282455, 74.9719315, -175.7859192, 174.9967346
5: -90.4393463, 70.3454666, -87.4257812, 68.0162354, -158.4555817, 157.7712402
6: -86.6970139, 83.5235748, -83.7994003, 80.7310104, -167.4280243, 167.3229523
7: -94.8713150, 79.9647064, -91.6993408, 77.3193970, -172.1907043, 171.6640472
8: -114.1304474, 77.2884140, -110.3076019, 74.6895065, -188.8199463, 187.5960083
9: -86.4736252, 84.5179214, -83.6045609, 81.6996155, -168.1732483, 168.1224670

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9789140, upper bound: 173.9833122
time: 8.56 seconds

## Relational analysis of IS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9961766, upper bound: 174.0059599
time: 8.20 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9961766, upper bound: 174.0059599
time: 7.78 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -93.8047333, 74.5136337, -105.6991653, 83.9540863, -177.7588196, 180.2127838
1: -78.6104126, 66.1015701, -88.4945602, 74.4153595, -153.0257721, 154.5961304
2: -103.5269089, 67.5729523, -116.5678711, 75.9626389, -179.4895020, 184.1408234
3: -109.8346786, 57.7635460, -123.7401199, 64.9252014, -174.7598724, 181.5036469
4: -100.3386002, 77.2042694, -113.1579437, 86.9111328, -187.2497253, 190.3622131
5: -90.0162277, 70.0184250, -101.4842300, 78.8699799, -168.8862000, 171.5026550
6: -86.2906189, 83.1322250, -97.2234116, 93.5800781, -179.8706818, 180.3556366
7: -94.4270630, 79.5938721, -106.2618103, 89.4792633, -183.9063263, 185.8556519
8: -113.5961609, 76.9242630, -127.8211594, 86.6612167, -200.2573853, 204.7454224
9: -86.0707550, 84.1223526, -96.8313675, 94.7412720, -180.8120270, 180.9537201

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9683517, upper bound: 173.9727849
time: 7.60 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9803164, upper bound: 173.9852913
time: 7.83 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9902199, upper bound: 174.0006379
time: 8.59 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9961766, upper bound: 174.0059599
time: 8.44 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -97.1398468, 77.0113602, -90.7215424, 72.0792313, -169.2190704, 167.7329102
1: -81.2423706, 68.2438812, -76.0174026, 63.9237099, -145.1660461, 144.2612457
2: -106.9815216, 69.6908875, -100.1197586, 65.3633423, -172.3448639, 169.8106384
3: -113.4796600, 59.5425911, -106.2027740, 55.8588066, -169.3384705, 165.7453613
4: -103.7134476, 79.6397400, -97.0255966, 74.6631927, -178.3766479, 176.6653290
5: -93.0929413, 72.0213623, -87.0678101, 67.7331161, -160.8260498, 159.0891724
6: -89.2468872, 85.9368439, -83.4551010, 80.3997421, -169.6466370, 169.3919373
7: -97.5071945, 82.1068726, -91.3225555, 77.0023117, -174.5094910, 173.4294281
8: -117.5732346, 79.3855209, -109.8581161, 74.3819275, -191.9551697, 189.2436218
9: -88.7992706, 86.7623596, -83.2602463, 81.3599319, -170.1592102, 170.0225830

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9765027, upper bound: 173.9765027
time: 6.16 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9765027, upper bound: 173.9765026
time: 6.02 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -96.6947327, 76.6595764, -105.3385086, 83.6654282, -180.3601685, 181.9980774
1: -80.8675308, 67.9294968, -88.1899414, 74.1575165, -155.0250397, 156.1194458
2: -106.4905930, 69.3725204, -116.1664810, 75.7013016, -182.1918640, 185.5390015
3: -112.9544067, 59.2675400, -123.3099976, 64.6996765, -177.6540680, 182.5775146
4: -103.2342072, 79.2723923, -112.7657852, 86.6104813, -189.8446503, 192.0381622
5: -92.6660767, 71.6914215, -101.1356430, 78.5941772, -171.2602539, 172.8270569
6: -88.8367233, 85.5421829, -96.8880310, 93.2575226, -182.0942383, 182.4302063
7: -97.0593338, 81.7328491, -105.8947983, 89.1704407, -186.2297668, 187.6276245
8: -117.0344086, 79.0183868, -127.3832703, 86.3616791, -203.3960724, 206.4016571
9: -88.3930283, 86.3632050, -96.4960098, 94.4105377, -182.8035583, 182.8592224

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9703843, upper bound: 173.9713904
time: 6.22 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9765027, upper bound: 173.9765026
time: 6.54 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 46.61 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 46.61
Output dim: 7, lower bound: -173.9961766, upper bound: 174.0059599
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 46.61
Output dim: 7, lower bound: -173.9961766, upper bound: 174.0059599
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 46.61
Output dim: 7, lower bound: -173.9902199, upper bound: 174.0006379
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 46.61
Output dim: 7, lower bound: -173.9961766, upper bound: 174.0059599
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 46.61
Output dim: 7, lower bound: -173.9765027, upper bound: 173.9765027
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 46.61
Output dim: 7, lower bound: -173.9765027, upper bound: 173.9765026
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 46.61
Output dim: 7, lower bound: -173.9703843, upper bound: 173.9713904
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 46.61
Output dim: 7, lower bound: -173.9765027, upper bound: 173.9765026

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -90.8334122, 72.1690445, -91.0917587, 72.3756943, -163.2091064, 163.2607727
1: -76.1121597, 64.0041580, -76.3301773, 64.1884613, -140.3006134, 140.3343048
2: -100.2447052, 65.4450836, -100.5319138, 65.6316681, -165.8763428, 165.9769897
3: -106.3366089, 55.9291992, -106.6444931, 56.0903130, -162.4269257, 162.5737000
4: -97.1475983, 74.7571487, -97.4282455, 74.9719315, -172.1195374, 172.1853943
5: -87.1761780, 67.8195038, -87.4257812, 68.0162354, -155.1924133, 155.2452850
6: -83.5594559, 80.5003052, -83.7994003, 80.7310104, -164.2904205, 164.2996826
7: -91.4370575, 77.0987625, -91.6993408, 77.3193970, -168.7564545, 168.7980957
8: -109.9940720, 74.4753036, -110.3076019, 74.6895065, -184.6835632, 184.7828979
9: -83.3649826, 81.4634323, -83.6045609, 81.6996155, -165.0645905, 165.0679626

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9789140, upper bound: 173.9833122
time: 8.96 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9884422, upper bound: 173.9927323
time: 8.30 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9495565, upper bound: 173.9490402
time: 9.20 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9633347, upper bound: 173.9639932
time: 7.14 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0013700, upper bound: 174.0093777
time: 8.02 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0038496, upper bound: 174.0126134
time: 8.65 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -105.4372940, 83.7445374, -91.0917587, 72.3756943, -177.8129578, 174.8362732
1: -88.2734985, 74.2285385, -76.3301773, 64.1884613, -152.4619598, 150.5586853
2: -116.2766953, 75.7733078, -100.5319138, 65.6316681, -181.9083557, 176.3052216
3: -123.4280319, 64.7618713, -106.6444931, 56.0903130, -179.5183411, 171.4063721
4: -112.8734436, 86.6933899, -97.4282455, 74.9719315, -187.8453674, 184.1216431
5: -101.2312012, 78.6704330, -87.4257812, 68.0162354, -169.2474213, 166.0962219
6: -96.9802322, 93.3461151, -83.7994003, 80.7310104, -177.7112274, 177.1455078
7: -105.9958954, 89.2556381, -91.6993408, 77.3193970, -183.3152771, 180.9549866
8: -127.5032578, 86.4439621, -110.3076019, 74.6895065, -202.1927490, 196.7515564
9: -96.5884933, 94.5018539, -83.6045609, 81.6996155, -178.2881165, 178.1064148

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9789140, upper bound: 173.9833122
time: 8.52 seconds

## Relational analysis of IS_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0013700, upper bound: 174.0093777
time: 8.93 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0038496, upper bound: 174.0126134
time: 8.01 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -93.3314819, 74.1370544, -100.4557114, 79.7988586, -173.1302948, 174.5927734
1: -78.2140808, 65.7676773, -84.1139374, 70.7182770, -148.9323578, 149.8816223
2: -103.0051117, 67.2388153, -110.8021317, 72.2673111, -175.2724152, 178.0409546
3: -109.2759476, 57.4709587, -117.5643845, 61.6768341, -170.9527588, 175.0353394
4: -99.8310928, 76.8155594, -107.5459290, 82.6030884, -182.4341736, 184.3614807
5: -89.5597763, 69.6667786, -96.4361267, 74.9715652, -164.5313263, 166.1028595
6: -85.8552704, 82.7131577, -92.3894424, 88.9375305, -174.7928009, 175.1026001
7: -93.9520721, 79.1958771, -101.0104370, 85.0789337, -179.0309906, 180.2063141
8: -113.0260925, 76.5376205, -121.5136795, 82.3673401, -195.3934326, 198.0512848
9: -85.6391373, 83.7001419, -92.0493546, 90.0487137, -175.6878510, 175.7494965

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9594256, upper bound: 173.9636817
time: 7.80 seconds

## Relational analysis of IS_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9736991, upper bound: 173.9791451
time: 7.84 seconds

## Relational analysis of IS_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=175.32177734375
rel_dist={7: [-174.07406456819004, 174.07406456764704]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9962271, upper bound: 174.0008380
time: 8.97 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9869953, upper bound: 173.9869953
time: 6.16 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.28 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.28
Output dim: 7, lower bound: -173.9962271, upper bound: 174.0008380
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.28
Output dim: 7, lower bound: -173.9869953, upper bound: 173.9869953

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -94.2460861, 74.8623047, -94.5060349, 75.0702057, -169.3162842, 169.3683319
1: -78.9821320, 66.4131470, -79.2014389, 66.5985794, -145.5807037, 145.6145935
2: -104.0140457, 67.8887329, -104.3030472, 68.0764999, -172.0905304, 172.1917725
3: -110.3552094, 58.0360298, -110.6649246, 58.1981163, -168.5533295, 168.7009277
4: -100.8139877, 77.5684891, -101.0963440, 77.7846146, -178.5986023, 178.6648254
5: -90.4393463, 70.3454666, -90.6905060, 70.5433807, -160.9827271, 161.0359802
6: -86.6970139, 83.5235748, -86.9384842, 83.7556839, -170.4526825, 170.4620209
7: -94.8713150, 79.9647064, -95.1351624, 80.1866226, -175.0579376, 175.0998688
8: -114.1304474, 77.2884140, -114.4460297, 77.5040588, -191.6344910, 191.7344360
9: -86.4736252, 84.5179214, -86.7146835, 84.7555695, -171.2291870, 171.2326050

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9760411, upper bound: 173.9775508
time: 9.02 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9939648, upper bound: 173.9980546
time: 8.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 8.66 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -97.1398468, 77.0113602, -93.4849319, 74.2527542, -171.3925629, 170.4962921
1: -81.2423706, 68.2438812, -78.3389893, 65.8683929, -147.1107635, 146.5828705
2: -106.9815216, 69.6908875, -103.1663513, 67.3365936, -174.3181152, 172.8572235
3: -113.4796600, 59.5425911, -109.4469452, 57.5597343, -171.0393677, 168.9895325
4: -103.7134476, 79.6397400, -99.9858551, 76.9331055, -180.6465302, 179.6255798
5: -93.0929413, 72.0213623, -89.7031631, 69.7627335, -162.8556671, 161.7245178
6: -89.2468872, 85.9368439, -85.9888000, 82.8421173, -172.0890045, 171.9256287
7: -97.5071945, 82.1068726, -94.0961227, 79.3124466, -176.8196259, 176.2030029
8: -117.5732346, 79.3855209, -113.2059174, 76.6553879, -194.2286224, 192.5914307
9: -88.7992706, 86.7623596, -85.7651672, 83.8187866, -172.6180573, 172.5275116

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9785582, upper bound: 173.9776385
time: 7.15 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
time: 5.95 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 35.45 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 35.45
Output dim: 7, lower bound: -173.9939648, upper bound: 173.9980546
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 35.45
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 35.45
Output dim: 7, lower bound: -173.9785582, upper bound: 173.9776385
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 35.45
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -93.1831741, 74.0235596, -91.0917587, 72.3756943, -165.5588684, 165.1153107
1: -78.0881729, 65.6629715, -76.3301773, 64.1884613, -142.2766113, 141.9931335
2: -102.8402481, 67.1276932, -100.5319138, 65.6316681, -168.4719086, 167.6596069
3: -109.1036530, 57.3801155, -106.6444931, 56.0903130, -165.1939697, 164.0246124
4: -99.6721725, 76.6927032, -97.4282455, 74.9719315, -174.6441040, 174.1209412
5: -89.4221573, 69.5592346, -87.4257812, 68.0162354, -157.4383850, 156.9850159
6: -85.7197876, 82.5819702, -83.7994003, 80.7310104, -166.4508057, 166.3813629
7: -93.8023300, 79.0722809, -91.6993408, 77.3193970, -171.1217041, 170.7716217
8: -112.8417816, 76.4119949, -110.3076019, 74.6895065, -187.5312805, 186.7196045
9: -85.5054398, 83.5662079, -83.6045609, 81.6996155, -167.2050476, 167.1707611

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 9.49 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 8.25 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -92.7955475, 73.7162628, -105.6991653, 83.9540863, -176.7496338, 179.4153748
1: -77.7604523, 65.3890305, -88.4945602, 74.4153595, -152.1758118, 153.8835754
2: -102.4129868, 66.8508759, -116.5678711, 75.9626389, -178.3755951, 183.4187469
3: -108.6444321, 57.1402359, -123.7401199, 64.9252014, -173.5696411, 180.8803558
4: -99.2515335, 76.3713913, -113.1579437, 86.9111328, -186.1626587, 189.5293274
5: -89.0489426, 69.2704544, -101.4842300, 78.8699799, -167.9188995, 170.7546844
6: -85.3612442, 82.2373886, -97.2234116, 93.5800781, -178.9413147, 179.4608002
7: -93.4111328, 78.7459106, -106.2618103, 89.4792633, -182.8903961, 185.0076752
8: -112.3742676, 76.0912933, -127.8211594, 86.6612167, -199.0354767, 203.9124451
9: -85.1495361, 83.2177429, -96.8313675, 94.7412720, -179.8908081, 180.0491028

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 8.09 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 8.87 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -96.0779190, 76.1727448, -90.0792007, 71.5648804, -167.6427917, 166.2519531
1: -80.3486328, 67.4941788, -75.4747543, 63.4642639, -143.8128815, 142.9689178
2: -105.8093796, 68.9302673, -99.4046555, 64.8977280, -170.7070923, 168.3349152
3: -112.2275467, 58.8868752, -105.4364014, 55.4570618, -167.6845856, 164.3232727
4: -102.5730591, 78.7647552, -96.3269577, 74.1274948, -176.7005310, 175.0917053
5: -92.0758133, 71.2353439, -86.4467239, 67.2417755, -159.3175812, 157.6820679
6: -88.2704697, 84.9959488, -82.8577499, 79.8249359, -168.0953979, 167.8536987
7: -96.4392624, 81.2143250, -90.6687317, 76.4521484, -172.8913879, 171.8830566
8: -116.2873611, 78.5121002, -109.0782318, 73.8483047, -190.1356506, 187.5903015
9: -87.8318253, 85.8111038, -82.6627808, 80.7705231, -168.6023254, 168.4738770

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
time: 6.60 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
time: 6.49 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -95.6868286, 75.8625793, -104.7123032, 83.1642075, -178.8510437, 180.5748901
1: -80.0181351, 67.2176895, -87.6610413, 73.7097702, -153.7279053, 154.8787231
2: -105.3788452, 68.6516953, -115.4694977, 75.2475204, -180.6263733, 184.1211853
3: -111.7636566, 58.6447220, -122.5632095, 64.3080902, -176.0717468, 181.2079315
4: -102.1490784, 78.4407272, -112.0848312, 86.0884171, -188.2374878, 190.5255585
5: -91.6992874, 70.9437561, -100.5304031, 78.1152420, -169.8145294, 171.4741516
6: -87.9084625, 84.6485291, -96.3057022, 92.6974335, -180.6058960, 180.9542236
7: -96.0447388, 80.8852386, -105.2575607, 88.6342392, -184.6789856, 186.1427917
8: -115.8163528, 78.1894073, -126.6229782, 85.8415298, -201.6578827, 204.8123779
9: -87.4729233, 85.4597778, -95.9136963, 93.8362885, -181.3092041, 181.3734131

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
time: 5.50 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
time: 6.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.69 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.69
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.69
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.69
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.69
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.69
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.69
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.69
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.69
Output dim: 7, lower bound: -173.9762404, upper bound: 173.9762404

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -90.8334122, 72.1690445, -91.0917587, 72.3756943, -163.2091064, 163.2607727
1: -76.1121597, 64.0041580, -76.3301773, 64.1884613, -140.3006134, 140.3343048
2: -100.2447052, 65.4450836, -100.5319138, 65.6316681, -165.8763428, 165.9769897
3: -106.3366089, 55.9291992, -106.6444931, 56.0903130, -162.4269257, 162.5737000
4: -97.1475983, 74.7571487, -97.4282455, 74.9719315, -172.1195374, 172.1853943
5: -87.1761780, 67.8195038, -87.4257812, 68.0162354, -155.1924133, 155.2452850
6: -83.5594559, 80.5003052, -83.7994003, 80.7310104, -164.2904205, 164.2996826
7: -91.4370575, 77.0987625, -91.6993408, 77.3193970, -168.7564545, 168.7980957
8: -109.9940720, 74.4753036, -110.3076019, 74.6895065, -184.6835632, 184.7828979
9: -83.3649826, 81.4634323, -83.6045609, 81.6996155, -165.0645905, 165.0679626

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9719896, upper bound: 173.9736990
time: 8.32 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9921561, upper bound: 173.9962535
time: 7.89 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9939648, upper bound: 173.9980547
time: 9.88 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -105.4372940, 83.7445374, -91.0917587, 72.3756943, -177.8129578, 174.8362732
1: -88.2734985, 74.2285385, -76.3301773, 64.1884613, -152.4619598, 150.5586853
2: -116.2766953, 75.7733078, -100.5319138, 65.6316681, -181.9083557, 176.3052216
3: -123.4280319, 64.7618713, -106.6444931, 56.0903130, -179.5183411, 171.4063721
4: -112.8734436, 86.6933899, -97.4282455, 74.9719315, -187.8453674, 184.1216431
5: -101.2312012, 78.6704330, -87.4257812, 68.0162354, -169.2474213, 166.0962219
6: -96.9802322, 93.3461151, -83.7994003, 80.7310104, -177.7112274, 177.1455078
7: -105.9958954, 89.2556381, -91.6993408, 77.3193970, -183.3152771, 180.9549866
8: -127.5032578, 86.4439621, -110.3076019, 74.6895065, -202.1927490, 196.7515564
9: -96.5884933, 94.5018539, -83.6045609, 81.6996155, -178.2881165, 178.1064148

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9934280, upper bound: 173.9977270
time: 8.26 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9939648, upper bound: 173.9980546
time: 9.04 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -90.8334122, 72.1690445, -105.6991653, 83.9540863, -174.7875061, 177.8681641
1: -76.1121597, 64.0041580, -88.4945602, 74.4153595, -150.5275269, 152.4987183
2: -100.2447052, 65.4450836, -116.5678711, 75.9626389, -176.2072906, 182.0129547
3: -106.3366089, 55.9291992, -123.7401199, 64.9252014, -171.2618103, 179.6693115
4: -97.1475983, 74.7571487, -113.1579437, 86.9111328, -184.0587311, 187.9150848
5: -87.1761780, 67.8195038, -101.4842300, 78.8699799, -166.0461578, 169.3037415
6: -83.5594559, 80.5003052, -97.2234116, 93.5800781, -177.1394958, 177.7237091
7: -91.4370575, 77.0987625, -106.2618103, 89.4792633, -180.9163208, 183.3605652
8: -109.9940720, 74.4753036, -127.8211594, 86.6612167, -196.6552887, 202.2964630
9: -83.3649826, 81.4634323, -96.8313675, 94.7412720, -178.1062317, 178.2947845

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9624625, upper bound: 173.9636519
time: 8.87 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9800736, upper bound: 173.9855334
time: 9.11 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 7.57 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -105.4372940, 83.7445374, -105.6991653, 83.9540863, -189.3913574, 189.4436646
1: -88.2734985, 74.2285385, -88.4945602, 74.4153595, -162.6888580, 162.7230835
2: -116.2766953, 75.7733078, -116.5678711, 75.9626389, -192.2393036, 192.3411865
3: -123.4280319, 64.7618713, -123.7401199, 64.9252014, -188.3532410, 188.5019836
4: -112.8734436, 86.6933899, -113.1579437, 86.9111328, -199.7845764, 199.8513184
5: -101.2312012, 78.6704330, -101.4842300, 78.8699799, -180.1011658, 180.1546631
6: -96.9802322, 93.3461151, -97.2234116, 93.5800781, -190.5602875, 190.5695190
7: -105.9958954, 89.2556381, -106.2618103, 89.4792633, -195.4751587, 195.5174561
8: -127.5032578, 86.4439621, -127.8211594, 86.6612167, -214.1644592, 214.2651215
9: -96.5884933, 94.5018539, -96.8313675, 94.7412720, -191.3297729, 191.3332214

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9624625, upper bound: 173.9636519
time: 9.93 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9800736, upper bound: 173.9855333
time: 9.67 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9867920, upper bound: 173.9918835
time: 8.49 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -93.7403030, 74.3259430, -90.0792007, 71.5648804, -165.3051758, 164.4051514
1: -78.3804855, 65.8434143, -75.4747543, 63.4642639, -141.8447571, 141.3181458
2: -103.2282486, 67.2559814, -99.4046555, 64.8977280, -168.1259613, 166.6606140
3: -109.4676819, 57.4434204, -105.4364014, 55.4570618, -164.9247284, 162.8798218
4: -100.0625229, 76.8396683, -96.3269577, 74.1274948, -174.1900024, 173.1666260
5: -89.8383331, 69.5029449, -86.4467239, 67.2417755, -157.0800781, 155.9496765
6: -86.1222763, 82.9249039, -82.8577499, 79.8249359, -165.9471741, 165.7826538
7: -94.0857697, 79.2475433, -90.6687317, 76.4521484, -170.5378876, 169.9162750
8: -113.4624100, 76.5956497, -109.0782318, 73.8483047, -187.3107147, 185.6738892
9: -85.7015686, 83.7190170, -82.6627808, 80.7705231, -166.4720917, 166.3817749

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9754658, upper bound: 173.9745976
time: 7.48 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9785582, upper bound: 173.9776385
time: 6.87 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -108.2859344, 85.8609009, -90.0792007, 71.5648804, -179.8508148, 175.9400940
1: -90.4996490, 76.0357590, -75.4747543, 63.4642639, -153.9638977, 151.5104828
2: -119.2036591, 77.5540924, -99.4046555, 64.8977280, -184.1013641, 176.9587402
3: -126.5062027, 66.2461929, -105.4364014, 55.4570618, -181.9632263, 171.6825867
4: -115.7342758, 88.7373047, -96.3269577, 74.1274948, -189.8617706, 185.0642548
5: -103.8441162, 80.3245316, -86.4467239, 67.2417755, -171.0858765, 166.7712402
6: -99.4937592, 95.7221069, -82.8577499, 79.8249359, -179.3186646, 178.5798492
7: -108.5926208, 91.3672485, -90.6687317, 76.4521484, -185.0447235, 182.0359802
8: -130.9031067, 88.5163498, -109.0782318, 73.8483047, -204.7514038, 197.5945740
9: -98.8762436, 96.7144318, -82.6627808, 80.7705231, -179.6467590, 179.3771973

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9716812, upper bound: 173.9705325
time: 7.39 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9785582, upper bound: 173.9776385
time: 7.56 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -93.7403030, 74.3259430, -104.7123032, 83.1642075, -176.9045105, 179.0382385
1: -78.3804855, 65.8434143, -87.6610413, 73.7097702, -152.0902557, 153.5044556
2: -103.2282486, 67.2559814, -115.4694977, 75.2475204, -178.4757690, 182.7254791
3: -109.4676819, 57.4434204, -122.5632095, 64.3080902, -173.7757721, 180.0066223
4: -100.0625229, 76.8396683, -112.0848312, 86.0884171, -186.1509247, 188.9244995
5: -89.8383331, 69.5029449, -100.5304031, 78.1152420, -167.9535522, 170.0333557
6: -86.1222763, 82.9249039, -96.3057022, 92.6974335, -178.8197021, 179.2306061
7: -94.0857697, 79.2475433, -105.2575607, 88.6342392, -182.7199860, 184.5050964
8: -113.4624100, 76.5956497, -126.6229782, 85.8415298, -199.3039398, 203.2186279
9: -85.7015686, 83.7190170, -95.9136963, 93.8362885, -179.5378571, 179.6326752

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=175.32177734375
rel_dist={7: [-174.07363473064066, 174.07363473064066]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9898603, upper bound: 173.9915006
time: 12.23 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9866161, upper bound: 173.9866162
time: 6.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.07 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.07
Output dim: 7, lower bound: -173.9898603, upper bound: 173.9915006
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.07
Output dim: 7, lower bound: -173.9866161, upper bound: 173.9866162

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -94.2460861, 74.8623047, -94.5060349, 75.0702057, -169.3162842, 169.3683319
1: -78.9821320, 66.4131470, -79.2014389, 66.5985794, -145.5807037, 145.6145935
2: -104.0140457, 67.8887329, -104.3030472, 68.0764999, -172.0905304, 172.1917725
3: -110.3552094, 58.0360298, -110.6649246, 58.1981163, -168.5533295, 168.7009277
4: -100.8139877, 77.5684891, -101.0963440, 77.7846146, -178.5986023, 178.6648254
5: -90.4393463, 70.3454666, -90.6905060, 70.5433807, -160.9827271, 161.0359802
6: -86.6970139, 83.5235748, -86.9384842, 83.7556839, -170.4526825, 170.4620209
7: -94.8713150, 79.9647064, -95.1351624, 80.1866226, -175.0579376, 175.0998688
8: -114.1304474, 77.2884140, -114.4460297, 77.5040588, -191.6344910, 191.7344360
9: -86.4736252, 84.5179214, -86.7146835, 84.7555695, -171.2291870, 171.2326050

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9839980, upper bound: 173.9852415
time: 11.51 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9794570, upper bound: 173.9812719
time: 10.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -97.1398468, 77.0113602, -91.3430481, 72.5379257, -169.6777649, 168.3544006
1: -81.2423706, 68.2438812, -76.5301285, 64.3364105, -145.5787659, 144.7739716
2: -106.9815216, 69.6908875, -100.7818527, 65.7842102, -172.7657318, 170.4726868
3: -113.4796600, 59.5425911, -106.8920517, 56.2201576, -169.6997986, 166.4346466
4: -103.7134476, 79.6397400, -97.6563797, 75.1467590, -178.8601990, 177.2961121
5: -93.0929413, 72.0213623, -87.6321640, 68.1248093, -161.2177429, 159.6535339
6: -89.2468872, 85.9368439, -83.9968033, 80.9255066, -170.1723938, 169.9336243
7: -97.5071945, 82.1068726, -91.9162827, 77.4784546, -174.9856567, 174.0231476
8: -117.5732346, 79.3855209, -110.6048355, 74.8751450, -192.4483490, 189.9903259
9: -88.7992706, 86.7623596, -83.7733002, 81.8537979, -170.6530762, 170.5356445

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9846935, upper bound: 173.9847377
time: 7.40 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9866162, upper bound: 173.9866161
time: 7.92 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 20.17 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 20.17
Output dim: 7, lower bound: -173.9839980, upper bound: 173.9852415
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 20.17
Output dim: 7, lower bound: -173.9794570, upper bound: 173.9812719
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 20.17
Output dim: 7, lower bound: -173.9846935, upper bound: 173.9847377
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 20.17
Output dim: 7, lower bound: -173.9866162, upper bound: 173.9866161

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -91.7543869, 72.8962097, -91.0917587, 72.3756943, -164.1300659, 163.9879608
1: -76.8872070, 64.6542130, -76.3301773, 64.1884613, -141.0756683, 140.9843750
2: -101.2620163, 66.1046600, -100.5319138, 65.6316681, -166.8936615, 166.6365509
3: -107.4223480, 56.4978600, -106.6444931, 56.0903130, -163.5126343, 163.1423492
4: -98.1368484, 75.5154266, -97.4282455, 74.9719315, -173.1087799, 172.9436646
5: -88.0564880, 68.5019150, -87.4257812, 68.0162354, -156.0727081, 155.9276886
6: -84.4055405, 81.3161697, -83.7994003, 80.7310104, -165.1365509, 165.1155548
7: -92.3645782, 77.8730316, -91.6993408, 77.3193970, -169.6839752, 169.5723724
8: -111.1081009, 75.2317352, -110.3076019, 74.6895065, -185.7976074, 185.5393372
9: -84.2040634, 82.2871323, -83.6045609, 81.6996155, -165.9036865, 165.8916626

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9794570, upper bound: 173.9812719
time: 11.11 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9794570, upper bound: 173.9812719
time: 11.73 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -91.6517868, 72.8126144, -105.6991653, 83.9540863, -175.6058655, 178.5117188
1: -76.7975006, 64.5812378, -88.4945602, 74.4153595, -151.2128601, 153.0757904
2: -101.1503601, 66.0325928, -116.5678711, 75.9626389, -177.1129913, 182.6004486
3: -107.2959290, 56.4332161, -123.7401199, 64.9252014, -172.2211304, 180.1733093
4: -98.0194016, 75.4274139, -113.1579437, 86.9111328, -184.9305115, 188.5853577
5: -87.9535294, 68.4225464, -101.4842300, 78.8699799, -166.8234863, 169.9067688
6: -84.3074646, 81.2232056, -97.2234116, 93.5800781, -177.8875427, 178.4465790
7: -92.2592545, 77.7850342, -106.2618103, 89.4792633, -181.7385254, 184.0468140
8: -110.9886475, 75.1460953, -127.8211594, 86.6612167, -197.6498566, 202.9672394
9: -84.1055756, 82.1925964, -96.8313675, 94.7412720, -178.8468475, 179.0239563

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 213

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9794570, upper bound: 173.9812719
time: 10.88 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9794570, upper bound: 173.9812719
time: 11.05 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -94.1133118, 74.6014709, -86.0416107, 68.3395462, -162.4528503, 160.6430359
1: -78.7068939, 66.1061249, -72.1044617, 60.6006851, -139.3075867, 138.2105713
2: -103.6439743, 67.5519485, -94.9574280, 62.0503769, -165.6943512, 162.5093536
3: -109.9016495, 57.6696396, -100.6513062, 52.9444847, -162.8461304, 158.3209534
4: -100.4648590, 77.1522827, -91.9834747, 70.7980118, -171.2628632, 169.1357574
5: -90.1707306, 69.7709808, -82.5318451, 64.1889191, -154.3596344, 152.3028259
6: -86.4575958, 83.2563248, -79.1127472, 76.2378845, -162.6954803, 162.3690338
7: -94.4667740, 79.5600586, -86.6112900, 73.0351257, -167.5018768, 166.1713562
8: -113.9268799, 76.9128189, -104.2359467, 70.5365143, -184.4633789, 181.1487579
9: -86.0378418, 84.0595627, -78.9465408, 77.1128159, -163.1506653, 163.0060730

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9845717, upper bound: 173.9845717
time: 9.23 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9845717, upper bound: 173.9847377
time: 8.54 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -95.7637482, 75.9172821, -89.4758987, 71.0552444, -166.8189850, 165.3931580
1: -80.0925903, 67.2742538, -74.9716949, 63.0226822, -143.1152649, 142.2459412
2: -105.4682465, 68.7188568, -98.7310791, 64.4669800, -169.9352264, 167.4499054
3: -111.8570251, 58.6925583, -104.6941986, 55.0681496, -166.9251709, 163.3867493
4: -102.2414169, 78.5108566, -95.6612854, 73.6171112, -175.8585205, 174.1721497
5: -91.7669449, 70.9968567, -85.8356628, 66.7351456, -158.5020905, 156.8325195
6: -87.9830856, 84.7210312, -82.2853622, 79.2776184, -167.2607117, 167.0063782
7: -96.1292419, 80.9502640, -90.0492859, 75.9104538, -172.0397034, 170.9995270
8: -115.9168701, 78.2617340, -108.3609467, 73.3507614, -189.2676392, 186.6226501
9: -87.5446625, 85.5366135, -82.0720139, 80.1939774, -167.7386475, 167.6086121

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9847377, upper bound: 173.9846935
time: 8.36 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9847377, upper bound: 173.9866162
time: 7.91 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 17.70 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.70
Output dim: 7, lower bound: -173.9794570, upper bound: 173.9812719
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.70
Output dim: 7, lower bound: -173.9794570, upper bound: 173.9812719
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.70
Output dim: 7, lower bound: -173.9794570, upper bound: 173.9812719
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.70
Output dim: 7, lower bound: -173.9794570, upper bound: 173.9812719
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.70
Output dim: 7, lower bound: -173.9845717, upper bound: 173.9845717
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.70
Output dim: 7, lower bound: -173.9845717, upper bound: 173.9847377
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.70
Output dim: 7, lower bound: -173.9847377, upper bound: 173.9846935
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.70
Output dim: 7, lower bound: -173.9847377, upper bound: 173.9866162

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -90.8334122, 72.1690445, -91.0917587, 72.3756943, -163.2091064, 163.2607727
1: -76.1121597, 64.0041580, -76.3301773, 64.1884613, -140.3006134, 140.3343048
2: -100.2447052, 65.4450836, -100.5319138, 65.6316681, -165.8763428, 165.9769897
3: -106.3366089, 55.9291992, -106.6444931, 56.0903130, -162.4269257, 162.5737000
4: -97.1475983, 74.7571487, -97.4282455, 74.9719315, -172.1195374, 172.1853943
5: -87.1761780, 67.8195038, -87.4257812, 68.0162354, -155.1924133, 155.2452850
6: -83.5594559, 80.5003052, -83.7994003, 80.7310104, -164.2904205, 164.2996826
7: -91.4370575, 77.0987625, -91.6993408, 77.3193970, -168.7564545, 168.7980957
8: -109.9940720, 74.4753036, -110.3076019, 74.6895065, -184.6835632, 184.7828979
9: -83.3649826, 81.4634323, -83.6045609, 81.6996155, -165.0645905, 165.0679626

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9811717, upper bound: 173.9822698
time: 11.46 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9839980, upper bound: 173.9852415
time: 12.88 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -105.4372940, 83.7445374, -91.0917587, 72.3756943, -177.8129578, 174.8362732
1: -88.2734985, 74.2285385, -76.3301773, 64.1884613, -152.4619598, 150.5586853
2: -116.2766953, 75.7733078, -100.5319138, 65.6316681, -181.9083557, 176.3052216
3: -123.4280319, 64.7618713, -106.6444931, 56.0903130, -179.5183411, 171.4063721
4: -112.8734436, 86.6933899, -97.4282455, 74.9719315, -187.8453674, 184.1216431
5: -101.2312012, 78.6704330, -87.4257812, 68.0162354, -169.2474213, 166.0962219
6: -96.9802322, 93.3461151, -83.7994003, 80.7310104, -177.7112274, 177.1455078
7: -105.9958954, 89.2556381, -91.6993408, 77.3193970, -183.3152771, 180.9549866
8: -127.5032578, 86.4439621, -110.3076019, 74.6895065, -202.1927490, 196.7515564
9: -96.5884933, 94.5018539, -83.6045609, 81.6996155, -178.2881165, 178.1064148

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 213

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9822385, upper bound: 173.9835720
time: 9.96 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9839980, upper bound: 173.9852415
time: 11.96 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -90.8334122, 72.1690445, -105.6991653, 83.9540863, -174.7875061, 177.8681641
1: -76.1121597, 64.0041580, -88.4945602, 74.4153595, -150.5275269, 152.4987183
2: -100.2447052, 65.4450836, -116.5678711, 75.9626389, -176.2072906, 182.0129547
3: -106.3366089, 55.9291992, -123.7401199, 64.9252014, -171.2618103, 179.6693115
4: -97.1475983, 74.7571487, -113.1579437, 86.9111328, -184.0587311, 187.9150848
5: -87.1761780, 67.8195038, -101.4842300, 78.8699799, -166.0461578, 169.3037415
6: -83.5594559, 80.5003052, -97.2234116, 93.5800781, -177.1394958, 177.7237091
7: -91.4370575, 77.0987625, -106.2618103, 89.4792633, -180.9163208, 183.3605652
8: -109.9940720, 74.4753036, -127.8211594, 86.6612167, -196.6552887, 202.2964630
9: -83.3649826, 81.4634323, -96.8313675, 94.7412720, -178.1062317, 178.2947845

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 213

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9721509, upper bound: 173.9740245
time: 10.92 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9794570, upper bound: 173.9812719
time: 9.64 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -105.4372940, 83.7445374, -105.6991653, 83.9540863, -189.3913574, 189.4436646
1: -88.2734985, 74.2285385, -88.4945602, 74.4153595, -162.6888580, 162.7230835
2: -116.2766953, 75.7733078, -116.5678711, 75.9626389, -192.2393036, 192.3411865
3: -123.4280319, 64.7618713, -123.7401199, 64.9252014, -188.3532410, 188.5019836
4: -112.8734436, 86.6933899, -113.1579437, 86.9111328, -199.7845764, 199.8513184
5: -101.2312012, 78.6704330, -101.4842300, 78.8699799, -180.1011658, 180.1546631
6: -96.9802322, 93.3461151, -97.2234116, 93.5800781, -190.5602875, 190.5695190
7: -105.9958954, 89.2556381, -106.2618103, 89.4792633, -195.4751587, 195.5174561
8: -127.5032578, 86.4439621, -127.8211594, 86.6612167, -214.1644592, 214.2651215
9: -96.5884933, 94.5018539, -96.8313675, 94.7412720, -191.3297729, 191.3332214

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9721509, upper bound: 173.9740245
time: 11.35 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9794570, upper bound: 173.9812719
time: 11.70 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -91.6357346, 72.6431198, -86.0416107, 68.3395462, -159.9752808, 158.6846924
1: -76.6388397, 64.3548965, -72.1044617, 60.6006851, -137.2395020, 136.4593353
2: -100.9234695, 65.8036346, -94.9574280, 62.0503769, -162.9738464, 160.7610321
3: -106.9830627, 56.1371117, -100.6513062, 52.9444847, -159.9275360, 156.7884216
4: -97.8212814, 75.1131363, -91.9834747, 70.7980118, -168.6192932, 167.0966034
5: -87.7880859, 67.9314575, -82.5318451, 64.1889191, -151.9770050, 150.4633026
6: -84.1641235, 81.0635452, -79.1127472, 76.2378845, -160.4020081, 160.1762848
7: -91.9860153, 77.4844818, -86.6112900, 73.0351257, -165.0211182, 164.0957642
8: -110.9468918, 74.8724823, -104.2359467, 70.5365143, -181.4833984, 179.1084290
9: -83.7797623, 81.8232422, -78.9465408, 77.1128159, -160.8925781, 160.7697754

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9691885, upper bound: 173.9688197
time: 7.87 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9682048, upper bound: 173.9682048
time: 8.40 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -95.2160568, 75.4818420, -86.0416107, 68.3395462, -163.5556030, 161.5234375
1: -79.6350174, 66.8883514, -72.1044617, 60.6006851, -140.2357025, 138.9927979
2: -104.8659668, 68.3319626, -94.9574280, 62.0503769, -166.9163513, 163.2893677
3: -111.2112732, 58.3541908, -100.6513062, 52.9444847, -164.1557617, 159.0054932
4: -101.6556473, 78.0615311, -91.9834747, 70.7980118, -172.4536591, 170.0450134
5: -91.2392349, 70.5890121, -82.5318451, 64.1889191, -155.4281464, 153.1208496
6: -87.4801407, 84.2371674, -79.1127472, 76.2378845, -163.7180176, 163.3499146
7: -95.5808640, 80.4899139, -86.6112900, 73.0351257, -168.6159515, 167.1011963
8: -115.2576599, 77.8144836, -104.2359467, 70.5365143, -185.7941589, 182.0503998
9: -87.0453568, 85.0487595, -78.9465408, 77.1128159, -164.1581726, 163.9953003

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9688196, upper bound: 173.9692788
time: 7.92 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9682048, upper bound: 173.9687092
time: 7.61 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -91.6357346, 72.6431198, -89.4758987, 71.0552444, -162.6909637, 162.1190186
1: -76.6388397, 64.3548965, -74.9716949, 63.0226822, -139.6614990, 139.3265533
2: -100.9234695, 65.8036346, -98.7310791, 64.4669800, -165.3904419, 164.5346832
3: -106.9830627, 56.1371117, -104.6941986, 55.0681496, -162.0512085, 160.8312988
4: -97.8212814, 75.1131363, -95.6612854, 73.6171112, -171.4383850, 170.7744141
5: -87.7880859, 67.9314575, -85.8356628, 66.7351456, -154.5232239, 153.7671204
6: -84.1641235, 81.0635452, -82.2853622, 79.2776184, -163.4417419, 163.3489075
7: -91.9860153, 77.4844818, -90.0492859, 75.9104538, -167.8964691, 167.5337524
8: -110.9468918, 74.8724823, -108.3609467, 73.3507614, -184.2976532, 183.2333984
9: -83.7797623, 81.8232422, -82.0720139, 80.1939774, -163.9737244, 163.8952637

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9691885, upper bound: 173.9688977
time: 7.50 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9682048, upper bound: 173.9685116
time: 7.33 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -95.2160568, 75.4818420, -89.4758987, 71.0552444, -166.2713013, 164.9577332
1: -79.6350174, 66.8883514, -74.9716949, 63.0226822, -142.6576996, 141.8600006
2: -104.8659668, 68.3319626, -98.7310791, 64.4669800, -169.3329468, 167.0630188
3: -111.2112732, 58.3541908, -104.6941986, 55.0681496, -166.2794189, 163.0483704
4: -101.6556473, 78.0615311, -95.6612854, 73.6171112, -175.2727661, 173.7228088
5: -91.2392349, 70.5890121, -85.8356628, 66.7351456, -157.9743805, 156.4246826
6: -87.4801407, 84.2371674, -82.2853622, 79.2776184, -166.7577515, 166.5225220
7: -95.5808640, 80.4899139, -90.0492859, 75.9104538, -171.4913177, 170.5391693
8: -115.2576599, 77.8144836, -108.3609467, 73.3507614, -188.6084137, 186.1753693
9: -87.0453568, 85.0487595, -82.0720139, 80.1939774, -167.2393188, 167.1207733

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9691885, upper bound: 173.9763245
time: 8.85 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9682048, upper bound: 173.9757895
time: 7.96 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.40 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 7, lower bound: -173.9811717, upper bound: 173.9822698
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 7, lower bound: -173.9839980, upper bound: 173.9852415
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 7, lower bound: -173.9822385, upper bound: 173.9835720
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 7, lower bound: -173.9839980, upper bound: 173.9852415
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 7, lower bound: -173.9721509, upper bound: 173.9740245
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 7, lower bound: -173.9794570, upper bound: 173.9812719
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 7, lower bound: -173.9721509, upper bound: 173.9740245
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 7, lower bound: -173.9794570, upper bound: 173.9812719
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 7, lower bound: -173.9691885, upper bound: 173.9688197
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 7, lower bound: -173.9682048, upper bound: 173.9682048
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 7, lower bound: -173.9688196, upper bound: 173.9692788
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 7, lower bound: -173.9682048, upper bound: 173.9687092
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 7, lower bound: -173.9691885, upper bound: 173.9688977
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 7, lower bound: -173.9682048, upper bound: 173.9685116
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 7, lower bound: -173.9691885, upper bound: 173.9763245
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 7, lower bound: -173.9682048, upper bound: 173.9757895

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -87.9978104, 69.9167480, -85.8327560, 68.2132339, -156.2110291, 155.7495117
1: -73.7415390, 62.0055771, -71.9396057, 60.4833603, -134.2248688, 133.9451904
2: -97.1244507, 63.4465714, -94.7567902, 61.9295082, -159.0539551, 158.2033234
3: -102.9906235, 54.1769409, -100.4537582, 52.8393555, -155.8299866, 154.6307068
4: -94.1068649, 72.4315720, -91.8001785, 70.6590805, -164.7659454, 164.2317505
5: -84.4443130, 65.7121201, -82.3655396, 64.1124725, -148.5567627, 148.0776215
6: -80.9541855, 77.9933777, -78.9549484, 76.0818863, -157.0359955, 156.9483185
7: -88.5958099, 74.7164993, -86.4394531, 72.9127731, -161.5085754, 161.1559448
8: -106.5872879, 72.1625443, -103.9950714, 70.3880157, -176.9752960, 176.1576233
9: -80.7791061, 78.9403076, -78.8168488, 77.0000534, -157.7791290, 157.7571106

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9878246, upper bound: 173.9894350
time: 10.21 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9878246, upper bound: 173.9894350
time: 9.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -89.5119247, 71.1204300, -89.2600250, 70.9226837, -160.4346008, 160.3804474
1: -75.0101471, 63.0746956, -74.8030090, 62.9004250, -137.9105530, 137.8777008
2: -98.7950439, 64.5132904, -98.5229111, 64.3406143, -163.1356506, 163.0361633
3: -104.7812119, 55.1138420, -104.4889069, 54.9603119, -159.7415009, 159.6027527
4: -95.7357941, 73.6751328, -95.4715118, 73.4725418, -169.2083435, 169.1466217
5: -85.9048996, 66.8359146, -85.6643524, 66.6528397, -152.5577087, 152.5002747
6: -82.3481674, 79.3350372, -82.1212845, 79.1160812, -161.4642487, 161.4563293
7: -90.1158600, 75.9893188, -89.8689575, 75.7819595, -165.8977966, 165.8582306
8: -108.4087830, 73.3972931, -108.1104126, 73.1956482, -181.6044159, 181.5077057
9: -82.1606903, 80.2900467, -81.9355164, 80.0737457, -162.2344055, 162.2255554

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9888792, upper bound: 173.9905057
time: 10.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9888792, upper bound: 173.9914998
time: 10.40 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -100.1949768, 79.5902710, -88.2564621, 70.1236649, -170.3186340, 167.8466949
1: -83.8938675, 70.5323029, -73.9598083, 62.1901093, -146.0839539, 144.4921112
2: -110.5122833, 72.0789261, -97.4120255, 63.6333847, -174.1456604, 169.4909515
3: -117.2536697, 61.5142593, -103.2988892, 54.3382721, -171.5919495, 164.8131409
4: -107.2626877, 82.3863602, -94.3878479, 72.6466446, -179.9093018, 176.7742004
5: -96.1841736, 74.7729797, -84.6942139, 65.9091034, -162.0932617, 159.4671936
6: -92.1473007, 88.7046967, -81.1943970, 78.2243500, -170.3716431, 169.8990631
7: -100.7457275, 84.8563232, -88.8584366, 74.9374008, -175.6831055, 173.7147522
8: -121.1972733, 82.1510849, -106.9012222, 72.3770142, -193.5742798, 189.0523071
9: -91.8075790, 89.8104477, -81.0189972, 79.1767807, -170.9843597, 170.8294220

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9809504, upper bound: 173.9821390
time: 10.55 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9809504, upper bound: 173.9835714
time: 10.82 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -103.6005707, 82.2880325, -89.7703400, 71.3271484, -174.9277191, 172.0583649
1: -86.7420731, 72.9369888, -75.2282181, 63.2590675, -150.0011139, 148.1652069
2: -114.2624054, 74.4783173, -99.0823517, 64.6999435, -178.9623260, 173.5606537
3: -121.2664795, 63.6286545, -105.0891724, 55.2750130, -176.5414734, 168.7178345
4: -110.9112015, 85.1897202, -96.0165176, 73.8899994, -184.8011780, 181.2062378
5: -99.4648285, 77.3036957, -86.1545715, 67.0327148, -166.4975433, 163.4582367
6: -95.2970734, 91.7267914, -82.5881805, 79.5658112, -174.8628845, 174.3149567
7: -104.1606598, 87.7138138, -90.3782196, 76.2099762, -180.3706360, 178.0920410
8: -125.2995682, 84.9462662, -108.7224197, 73.6115799, -198.9111481, 193.6686401
9: -94.9145508, 92.8713989, -82.4003448, 80.5263214, -175.4408722, 175.2717438

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9811717, upper bound: 173.9822698
time: 11.10 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9811717, upper bound: 173.9852415
time: 11.02 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -87.9978104, 69.9167480, -100.4557114, 79.7988586, -167.7966156, 170.3724518
1: -73.7415390, 62.0055771, -84.1139374, 70.7182770, -144.4597778, 146.1195068
2: -97.1244507, 63.4465714, -110.8021317, 72.2673111, -169.3917542, 174.2486877
3: -102.9906235, 54.1769409, -117.5643845, 61.6768341, -164.6674500, 171.7413330
4: -94.1068649, 72.4315720, -107.5459290, 82.6030884, -176.7099609, 179.9775085
5: -84.4443130, 65.7121201, -96.4361267, 74.9715652, -159.4158478, 162.1482086
6: -80.9541855, 77.9933777, -92.3894424, 88.9375305, -169.8916931, 170.3828125
7: -88.5958099, 74.7164993, -101.0104370, 85.0789337, -173.6747437, 175.7269287
8: -106.5872879, 72.1625443, -121.5136795, 82.3673401, -188.9545898, 193.6762238
9: -80.7791061, 78.9403076, -92.0493546, 90.0487137, -170.8278046, 170.9896545

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 213

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9721756, upper bound: 173.9740617
time: 11.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9721756, upper bound: 173.9740616
time: 9.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -89.5119247, 71.1204300, -103.8624954, 82.4976349, -172.0095215, 174.9829254
1: -75.0101471, 63.0746956, -86.9631882, 73.1238937, -148.1340332, 150.0378876
2: -98.7950439, 64.5132904, -114.5536346, 74.6676941, -173.4627380, 179.0669250
3: -104.7812119, 55.1138420, -121.5786133, 63.7920189, -168.5732117, 176.6924591
4: -95.7357941, 73.6751328, -111.1957855, 85.4075165, -181.1433105, 184.8708801
5: -85.9048996, 66.8359146, -99.7179337, 77.5032578, -163.4081268, 166.5538483
6: -82.3481674, 79.3350372, -95.5403290, 91.9608231, -174.3089905, 174.8753662
7: -90.1158600, 75.9893188, -104.4266281, 87.9375076, -178.0533447, 180.4159393
8: -108.4087830, 73.3972931, -125.6175308, 85.1635666, -193.5723572, 199.0148315
9: -82.1606903, 80.2900467, -95.1574936, 93.1108627, -175.2715454, 175.4475098

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 213

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9776071, upper bound: 173.9793133
time: 9.51 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9776071, upper bound: 173.9813115
time: 10.19 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -102.5911560, 81.4848022, -100.4557114, 79.7988586, -182.3899841, 181.9405060
1: -85.8948212, 72.2234268, -84.1139374, 70.7182770, -156.6130676, 156.3373718
2: -113.1457443, 73.7677536, -110.8021317, 72.2673111, -185.4130554, 184.5698853
3: -120.0709763, 63.0035057, -117.5643845, 61.6768341, -181.7478027, 180.5678864
4: -109.8224792, 84.3601990, -107.5459290, 82.6030884, -192.4255676, 191.9061279
5: -98.4897690, 76.5555191, -96.4361267, 74.9715652, -173.4613190, 172.9916382
6: -94.3658142, 90.8304672, -92.3894424, 88.9375305, -183.3033447, 183.2199097
7: -103.1448975, 86.8652267, -101.0104370, 85.0789337, -188.2238007, 187.8756714
8: -124.0840683, 84.1234360, -121.5136795, 82.3673401, -206.4513855, 205.6371002
9: -93.9934006, 91.9698181, -92.0493546, 90.0487137, -184.0421143, 184.0191650

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9721508, upper bound: 173.9740245
time: 11.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9721508, upper bound: 173.9740245
time: 11.21 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -104.1169739, 82.6972961, -103.8624954, 82.4976349, -186.6145935, 186.5597687
1: -87.1723328, 73.2999268, -86.9631882, 73.1238937, -160.2962341, 160.2631073
2: -114.8284836, 74.8421402, -114.5536346, 74.6676941, -189.4961700, 189.3957825
3: -121.8738251, 63.9470978, -121.5786133, 63.7920189, -185.6658325, 185.5257111
4: -111.4627533, 85.6122437, -111.1957855, 85.4075165, -196.8702545, 196.8079987
5: -99.9610291, 77.6880035, -99.7179337, 77.5032578, -177.4642792, 177.4059448
6: -95.7698059, 92.1818619, -95.5403290, 91.9608231, -187.7306213, 187.7221985
7: -104.6761856, 88.1470718, -104.4266281, 87.9375076, -192.6136475, 192.5737000
8: -125.9190369, 85.3671112, -125.6175308, 85.1635666, -211.0826111, 210.9846344
9: -95.3850174, 93.3294601, -95.1574936, 93.1108627, -188.4958801, 188.4869080

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 73

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9775402, upper bound: 173.9792285
time: 10.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9775402, upper bound: 173.9812719
time: 11.01 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -89.1560898, 70.6848602, -82.6730118, 65.6830750, -154.8391724, 153.3578796
1: -74.5532684, 62.6059647, -69.2701950, 58.2225456, -132.7758026, 131.8761444
2: -98.1860046, 64.0292587, -91.2388306, 59.6381645, -157.8241730, 155.2680969
3: -104.0603180, 54.6127243, -96.6820602, 50.8627434, -154.9230652, 151.2947540
4: -95.1615295, 73.0722961, -88.3631821, 68.0229874, -163.1845093, 161.4354858
5: -85.4130020, 66.0994110, -79.3109818, 61.6953316, -147.1083374, 145.4104004
6: -81.8863602, 78.8697205, -76.0161591, 73.2535248, -155.1398468, 154.8858795
7: -89.4925613, 75.4021606, -83.2231369, 70.2049408, -159.6974945, 158.6252441
8: -107.9485626, 72.8400955, -100.1575012, 67.7624512, -175.7109680, 172.9975891
9: -81.5229111, 79.6048813, -75.8762283, 74.0986481, -155.6215515, 155.4811096

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 253
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9682050, upper bound: 173.9682048
time: 8.95 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9682050, upper bound: 173.9682050
time: 8.26 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 18.81 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.81
Output dim: 7, lower bound: -173.9878246, upper bound: 173.9894350
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.81
Output dim: 7, lower bound: -173.9878246, upper bound: 173.9894350
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.81
Output dim: 7, lower bound: -173.9888792, upper bound: 173.9905057
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.81
Output dim: 7, lower bound: -173.9888792, upper bound: 173.9914998
IS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 18.81
Output dim: 7, lower bound: -173.9809504, upper bound: 173.9821390
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 18.81
Output dim: 7, lower bound: -173.9809504, upper bound: 173.9835714
IS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.81
Output dim: 7, lower bound: -173.9811717, upper bound: 173.9822698
IS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 18.81
Output dim: 7, lower bound: -173.9811717, upper bound: 173.9852415
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.81
Output dim: 7, lower bound: -173.9721756, upper bound: 173.9740617
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.81
Output dim: 7, lower bound: -173.9721756, upper bound: 173.9740616
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.81
Output dim: 7, lower bound: -173.9776071, upper bound: 173.9793133
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.81
Output dim: 7, lower bound: -173.9776071, upper bound: 173.9813115
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.81
Output dim: 7, lower bound: -173.9721508, upper bound: 173.9740245
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.81
Output dim: 7, lower bound: -173.9721508, upper bound: 173.9740245
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.81
Output dim: 7, lower bound: -173.9775402, upper bound: 173.9792285
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.81
Output dim: 7, lower bound: -173.9775402, upper bound: 173.9812719
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.81
Output dim: 7, lower bound: -173.9682050, upper bound: 173.9682048
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.81
Output dim: 7, lower bound: -173.9682050, upper bound: 173.9682050
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -173.9682048, upper bound: 173.9682048
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -173.9688196, upper bound: 173.9692788
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -173.9682048, upper bound: 173.9687092
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -173.9691885, upper bound: 173.9688977
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -173.9682048, upper bound: 173.9685116
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -173.9691885, upper bound: 173.9763245
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -173.9682048, upper bound: 173.9757895
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=175.32177734375
rel_dist={7: [-174.07321147769613, 174.07321147870266]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1806.89 seconds
