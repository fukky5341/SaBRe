## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 193.149136115
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694)
1: (-90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682)
2: (-118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309)
3: (-125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304)
4: (-114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151)
5: (-102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109)
6: (-98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933)
7: (-107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075)
8: (-129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496)
9: (-97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215)

## BASE Result
execution time: IAR + LP analysis = 1.29 + 9.07 = 10.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -193.2890147, upper bound: 193.2890147


# Binary Search by BASE starts (time budget: 2689.64 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=194.859130859375
rel_dist={2: [-193.2889031662745, 193.28890316692934]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=194.859130859375
rel_dist={2: [-193.2888173876961, 193.28881738769616]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=194.859130859375
rel_dist={2: [-193.2885465354435, 193.28854653544352]}

## Binary Search Result
Binary search time: 35.84 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2653.79 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2857971, upper bound: 193.2858051
time: 5.80 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2858051, upper bound: 193.2857971
time: 5.89 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.83 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.83
Output dim: 2, lower bound: -193.2857971, upper bound: 193.2858051
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.83
Output dim: 2, lower bound: -193.2858051, upper bound: 193.2857971

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2701781, upper bound: 193.2701744
time: 5.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2701781, upper bound: 193.2701744
time: 5.61 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2701744, upper bound: 193.2701781
time: 5.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2701744, upper bound: 193.2701781
time: 5.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 12.88 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.88
Output dim: 2, lower bound: -193.2701781, upper bound: 193.2701744
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.88
Output dim: 2, lower bound: -193.2701781, upper bound: 193.2701744
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.88
Output dim: 2, lower bound: -193.2701744, upper bound: 193.2701781
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.88
Output dim: 2, lower bound: -193.2701744, upper bound: 193.2701781

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2541854, upper bound: 193.2541915
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2541854, upper bound: 193.2541915
time: 5.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2541854, upper bound: 193.2541915
time: 5.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2541854, upper bound: 193.2541915
time: 5.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2541915, upper bound: 193.2541854
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2541915, upper bound: 193.2541854
time: 5.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2541915, upper bound: 193.2541854
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2541915, upper bound: 193.2541854
time: 6.10 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.18 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.18
Output dim: 2, lower bound: -193.2541854, upper bound: 193.2541915
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.18
Output dim: 2, lower bound: -193.2541854, upper bound: 193.2541915
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.18
Output dim: 2, lower bound: -193.2541854, upper bound: 193.2541915
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.18
Output dim: 2, lower bound: -193.2541854, upper bound: 193.2541915
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.18
Output dim: 2, lower bound: -193.2541915, upper bound: 193.2541854
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.18
Output dim: 2, lower bound: -193.2541915, upper bound: 193.2541854
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.18
Output dim: 2, lower bound: -193.2541915, upper bound: 193.2541854
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.18
Output dim: 2, lower bound: -193.2541915, upper bound: 193.2541854

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689411, upper bound: 193.1689412
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689411, upper bound: 193.1689412
time: 4.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689411, upper bound: 193.1689412
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689411, upper bound: 193.1689412
time: 4.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689411, upper bound: 193.1689412
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689411, upper bound: 193.1689412
time: 4.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689411, upper bound: 193.1689412
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689411, upper bound: 193.1689412
time: 4.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689411
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689411
time: 5.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689411
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689411
time: 5.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689411
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689411
time: 5.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689411
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689411
time: 5.07 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 11.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.42
Output dim: 2, lower bound: -193.1689411, upper bound: 193.1689412
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.42
Output dim: 2, lower bound: -193.1689411, upper bound: 193.1689412
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.42
Output dim: 2, lower bound: -193.1689411, upper bound: 193.1689412
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.42
Output dim: 2, lower bound: -193.1689411, upper bound: 193.1689412
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.42
Output dim: 2, lower bound: -193.1689411, upper bound: 193.1689412
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.42
Output dim: 2, lower bound: -193.1689411, upper bound: 193.1689412
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.42
Output dim: 2, lower bound: -193.1689411, upper bound: 193.1689412
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.42
Output dim: 2, lower bound: -193.1689411, upper bound: 193.1689412
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.42
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689411
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.42
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689411
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.42
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689411
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.42
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689411
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.42
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689411
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.42
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689411
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.42
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689411
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.42
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689411

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689401
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689406, upper bound: 193.1689412
time: 4.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689401
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689406, upper bound: 193.1689412
time: 4.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689401
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689406, upper bound: 193.1689412
time: 5.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689401
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689406, upper bound: 193.1689412
time: 5.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689401
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689406, upper bound: 193.1689412
time: 4.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689401
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689406, upper bound: 193.1689412
time: 4.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689401
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689406, upper bound: 193.1689412
time: 4.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689401
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689406, upper bound: 193.1689412
time: 4.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
time: 5.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
time: 4.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
time: 5.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
time: 5.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
time: 4.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
time: 4.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
time: 5.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
time: 5.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689401
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689406, upper bound: 193.1689412
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689401
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689406, upper bound: 193.1689412
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689401
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689406, upper bound: 193.1689412
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689401
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689406, upper bound: 193.1689412
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689401
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689406, upper bound: 193.1689412
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689401
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689406, upper bound: 193.1689412
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689401
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689406, upper bound: 193.1689412
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689401
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689406, upper bound: 193.1689412
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
time: 4.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
time: 4.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
time: 4.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
time: 5.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
time: 4.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
time: 5.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
time: 4.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
time: 5.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
time: 4.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
time: 4.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
time: 5.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
time: 5.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
time: 5.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
time: 5.16 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 13.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495256
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.01
Output dim: 2, lower bound: -193.1495338, upper bound: 193.1495255
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 2, lower bound: -193.1689412, upper bound: 193.1689406
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 2, lower bound: -193.1689402, upper bound: 193.1689412
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=194.859130859375
rel_dist={2: [-193.2889031662745, 193.28890316692934]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2857393, upper bound: 193.2857449
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2857449, upper bound: 193.2857393
time: 7.03 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.29 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.29
Output dim: 2, lower bound: -193.2857393, upper bound: 193.2857449
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.29
Output dim: 2, lower bound: -193.2857449, upper bound: 193.2857393

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2700589, upper bound: 193.2700564
time: 6.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2700589, upper bound: 193.2700564
time: 6.72 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2700564, upper bound: 193.2700589
time: 7.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2700564, upper bound: 193.2700589
time: 6.98 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.52 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.52
Output dim: 2, lower bound: -193.2700589, upper bound: 193.2700564
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.52
Output dim: 2, lower bound: -193.2700589, upper bound: 193.2700564
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.52
Output dim: 2, lower bound: -193.2700564, upper bound: 193.2700589
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.52
Output dim: 2, lower bound: -193.2700564, upper bound: 193.2700589

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2541332, upper bound: 193.2541353
time: 7.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2541332, upper bound: 193.2541353
time: 6.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2541332, upper bound: 193.2541352
time: 6.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2541332, upper bound: 193.2541353
time: 7.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2541352, upper bound: 193.2541332
time: 6.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2541352, upper bound: 193.2541332
time: 7.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2541352, upper bound: 193.2541332
time: 6.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2541352, upper bound: 193.2541332
time: 5.84 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.71 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.71
Output dim: 2, lower bound: -193.2541332, upper bound: 193.2541353
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.71
Output dim: 2, lower bound: -193.2541332, upper bound: 193.2541353
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.71
Output dim: 2, lower bound: -193.2541332, upper bound: 193.2541352
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.71
Output dim: 2, lower bound: -193.2541332, upper bound: 193.2541353
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.71
Output dim: 2, lower bound: -193.2541352, upper bound: 193.2541332
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.71
Output dim: 2, lower bound: -193.2541352, upper bound: 193.2541332
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.71
Output dim: 2, lower bound: -193.2541352, upper bound: 193.2541332
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.71
Output dim: 2, lower bound: -193.2541352, upper bound: 193.2541332

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688679
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688679
time: 4.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
time: 5.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688679
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688679
time: 4.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
time: 5.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
time: 4.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
time: 5.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
time: 4.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
time: 5.11 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 11.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.66
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688679
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.66
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688679
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.66
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.66
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.66
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688679
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.66
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688679
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.66
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.66
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.66
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.66
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.66
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.66
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.66
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.66
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.66
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.66
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688680

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688623
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688626, upper bound: 193.1688680
time: 5.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688623
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688626, upper bound: 193.1688680
time: 5.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688623
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688626, upper bound: 193.1688680
time: 5.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688623
time: 5.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688626, upper bound: 193.1688680
time: 6.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688623
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688626, upper bound: 193.1688680
time: 5.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688623
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688626, upper bound: 193.1688680
time: 5.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688623
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688626, upper bound: 193.1688680
time: 5.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688623
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688626, upper bound: 193.1688680
time: 5.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
time: 5.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
time: 5.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
time: 5.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
time: 4.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
time: 5.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
time: 5.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
time: 5.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
time: 4.95 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688623
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688626, upper bound: 193.1688680
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688623
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688626, upper bound: 193.1688680
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688623
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688626, upper bound: 193.1688680
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688623
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688626, upper bound: 193.1688680
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688623
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688626, upper bound: 193.1688680
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688623
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688626, upper bound: 193.1688680
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688623
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688626, upper bound: 193.1688680
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688623
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688626, upper bound: 193.1688680
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
time: 5.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
time: 5.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
time: 5.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
time: 5.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
time: 5.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
time: 4.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
time: 5.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
time: 4.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
time: 5.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
time: 5.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
time: 5.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
time: 4.97 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 13.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493057, upper bound: 193.1493024
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.25
Output dim: 2, lower bound: -193.1493056, upper bound: 193.1493030
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688623
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688626, upper bound: 193.1688680
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688623
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688626, upper bound: 193.1688680
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688679, upper bound: 193.1688626
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.25
Output dim: 2, lower bound: -193.1688622, upper bound: 193.1688679
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=194.859130859375
rel_dist={2: [-193.2888173876961, 193.28881738769616]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2852996, upper bound: 193.2853007
time: 7.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2853007, upper bound: 193.2852996
time: 9.80 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.55 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.55
Output dim: 2, lower bound: -193.2852996, upper bound: 193.2853007
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.55
Output dim: 2, lower bound: -193.2853007, upper bound: 193.2852996

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2695610, upper bound: 193.2695609
time: 8.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2695610, upper bound: 193.2695609
time: 8.01 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2695609, upper bound: 193.2695610
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2695609, upper bound: 193.2695610
time: 7.64 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.37 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.37
Output dim: 2, lower bound: -193.2695610, upper bound: 193.2695609
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.37
Output dim: 2, lower bound: -193.2695610, upper bound: 193.2695609
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.37
Output dim: 2, lower bound: -193.2695609, upper bound: 193.2695610
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.37
Output dim: 2, lower bound: -193.2695609, upper bound: 193.2695610

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2537772, upper bound: 193.2537795
time: 7.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2537772, upper bound: 193.2537795
time: 7.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2537772, upper bound: 193.2537795
time: 8.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2537772, upper bound: 193.2537795
time: 7.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2537795, upper bound: 193.2537772
time: 8.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2537795, upper bound: 193.2537772
time: 8.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2537795, upper bound: 193.2537772
time: 8.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2537795, upper bound: 193.2537772
time: 8.52 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.32 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 2, lower bound: -193.2537772, upper bound: 193.2537795
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 2, lower bound: -193.2537772, upper bound: 193.2537795
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 2, lower bound: -193.2537772, upper bound: 193.2537795
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 2, lower bound: -193.2537772, upper bound: 193.2537795
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 2, lower bound: -193.2537795, upper bound: 193.2537772
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 2, lower bound: -193.2537795, upper bound: 193.2537772
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 2, lower bound: -193.2537795, upper bound: 193.2537772
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 2, lower bound: -193.2537795, upper bound: 193.2537772

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685554, upper bound: 193.1685552
time: 5.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685554, upper bound: 193.1685552
time: 5.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685554, upper bound: 193.1685552
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685554, upper bound: 193.1685552
time: 5.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685554, upper bound: 193.1685550
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685554, upper bound: 193.1685550
time: 5.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685554, upper bound: 193.1685550
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685554, upper bound: 193.1685550
time: 5.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685554
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685554
time: 6.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685554
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685554
time: 6.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
time: 5.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
time: 5.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
time: 5.78 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 12.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.94
Output dim: 2, lower bound: -193.1685554, upper bound: 193.1685552
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.94
Output dim: 2, lower bound: -193.1685554, upper bound: 193.1685552
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.94
Output dim: 2, lower bound: -193.1685554, upper bound: 193.1685552
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.94
Output dim: 2, lower bound: -193.1685554, upper bound: 193.1685552
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.94
Output dim: 2, lower bound: -193.1685554, upper bound: 193.1685550
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.94
Output dim: 2, lower bound: -193.1685554, upper bound: 193.1685550
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.94
Output dim: 2, lower bound: -193.1685554, upper bound: 193.1685550
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.94
Output dim: 2, lower bound: -193.1685554, upper bound: 193.1685550
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.94
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685554
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.94
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685554
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.94
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685554
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.94
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685554
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.94
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.94
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.94
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.94
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
time: 7.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
time: 7.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
time: 7.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
time: 7.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
time: 7.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
time: 7.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
time: 6.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
time: 6.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
time: 7.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
time: 7.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
time: 7.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
time: 7.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
time: 7.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
time: 5.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
time: 5.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
time: 5.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
time: 5.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
time: 5.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
time: 5.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
time: 5.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
time: 5.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
time: 5.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
time: 5.71 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 15.17 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.17
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1488662, upper bound: 193.1488637
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1488662, upper bound: 193.1488637
time: 6.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1488636, upper bound: 193.1488661
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1488636, upper bound: 193.1488661
time: 5.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1488662, upper bound: 193.1488637
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1488662, upper bound: 193.1488637
time: 6.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1488636, upper bound: 193.1488661
time: 6.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1488636, upper bound: 193.1488661
time: 5.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1488662, upper bound: 193.1488637
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1488662, upper bound: 193.1488637
time: 5.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1488636, upper bound: 193.1488661
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1488636, upper bound: 193.1488661
time: 4.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.9841232, 85.0895767, -106.9841232, 85.0895767, -192.0736694, 192.0736694
1: -90.1812515, 75.6211243, -90.1812515, 75.6211243, -165.8023682, 165.8023682
2: -118.1329117, 76.7262192, -118.1329117, 76.7262192, -194.8591309, 194.8591309
3: -125.3423386, 66.6745224, -125.3423386, 66.6745224, -192.0168304, 192.0168304
4: -114.6084290, 88.1751862, -114.6084290, 88.1751862, -202.7836151, 202.7836151
5: -102.6499710, 80.0399475, -102.6499710, 80.0399475, -182.6899109, 182.6899109
6: -98.4937134, 94.9893875, -98.4937134, 94.9893875, -193.4830933, 193.4830933
7: -107.6003113, 90.1861038, -107.6003113, 90.1861038, -197.7864075, 197.7864075
8: -129.7671661, 88.8023758, -129.7671661, 88.8023758, -218.5695496, 218.5695496
9: -97.6945114, 96.3775101, -97.6945114, 96.3775101, -194.0720215, 194.0720215

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1488662, upper bound: 193.1488637
time: 5.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1488662, upper bound: 193.1488637
time: 5.66 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 15.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.05
Output dim: 2, lower bound: -193.1488662, upper bound: 193.1488637
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.05
Output dim: 2, lower bound: -193.1488662, upper bound: 193.1488637
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.05
Output dim: 2, lower bound: -193.1488636, upper bound: 193.1488661
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.05
Output dim: 2, lower bound: -193.1488636, upper bound: 193.1488661
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.05
Output dim: 2, lower bound: -193.1488662, upper bound: 193.1488637
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.05
Output dim: 2, lower bound: -193.1488662, upper bound: 193.1488637
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.05
Output dim: 2, lower bound: -193.1488636, upper bound: 193.1488661
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.05
Output dim: 2, lower bound: -193.1488636, upper bound: 193.1488661
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.05
Output dim: 2, lower bound: -193.1488662, upper bound: 193.1488637
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.05
Output dim: 2, lower bound: -193.1488662, upper bound: 193.1488637
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.05
Output dim: 2, lower bound: -193.1488636, upper bound: 193.1488661
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.05
Output dim: 2, lower bound: -193.1488636, upper bound: 193.1488661
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.05
Output dim: 2, lower bound: -193.1488662, upper bound: 193.1488637
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.05
Output dim: 2, lower bound: -193.1488662, upper bound: 193.1488637
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685551
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685552, upper bound: 193.1685552
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685554
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 2, lower bound: -193.1685550, upper bound: 193.1685552
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=194.859130859375
rel_dist={2: [-193.2885465354435, 193.28854653544352]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1815.07 seconds
