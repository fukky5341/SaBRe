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
execution time: IAR + LP analysis = 1.24 + 9.03 = 10.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -193.2890147, upper bound: 193.2890147


# Binary Search by BASE starts (time budget: 2689.72 seconds, max iter: 100)

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
Binary search time: 35.60 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2654.12 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2411907, upper bound: 193.2411907
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2411907, upper bound: 193.2411907
time: 5.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.25 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.25
Output dim: 2, lower bound: -193.2411907, upper bound: 193.2411907
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.25
Output dim: 2, lower bound: -193.2411907, upper bound: 193.2411907

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
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2411602, upper bound: 193.2411602
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2411602, upper bound: 193.2411602
time: 4.85 seconds

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
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2359925, upper bound: 193.2359925
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2359925, upper bound: 193.2359925
time: 6.02 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.15 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.15
Output dim: 2, lower bound: -193.2411602, upper bound: 193.2411602
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.15
Output dim: 2, lower bound: -193.2411602, upper bound: 193.2411602
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.15
Output dim: 2, lower bound: -193.2359925, upper bound: 193.2359925
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.15
Output dim: 2, lower bound: -193.2359925, upper bound: 193.2359925

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2411457, upper bound: 193.2411602
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2411602, upper bound: 193.2411457
time: 5.20 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2136797, upper bound: 193.2136797
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2136797, upper bound: 193.2136797
time: 5.87 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2359097, upper bound: 193.2358605
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2358605, upper bound: 193.2359097
time: 5.05 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2308986, upper bound: 193.2308986
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2308986, upper bound: 193.2308986
time: 4.96 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 11.10 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.10
Output dim: 2, lower bound: -193.2411457, upper bound: 193.2411602
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.10
Output dim: 2, lower bound: -193.2411602, upper bound: 193.2411457
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.10
Output dim: 2, lower bound: -193.2136797, upper bound: 193.2136797
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.10
Output dim: 2, lower bound: -193.2136797, upper bound: 193.2136797
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.10
Output dim: 2, lower bound: -193.2359097, upper bound: 193.2358605
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.10
Output dim: 2, lower bound: -193.2358605, upper bound: 193.2359097
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.10
Output dim: 2, lower bound: -193.2308986, upper bound: 193.2308986
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.10
Output dim: 2, lower bound: -193.2308986, upper bound: 193.2308986

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2359509, upper bound: 193.2359925
time: 5.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2359509, upper bound: 193.2359925
time: 5.95 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1399371, upper bound: 193.1399375
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1399371, upper bound: 193.1399375
time: 4.36 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2074982, upper bound: 193.2074982
time: 7.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2074982, upper bound: 193.2074982
time: 6.08 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1535522, upper bound: 193.1535522
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1535522, upper bound: 193.1535522
time: 5.05 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2359086, upper bound: 193.2358605
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2359097, upper bound: 193.2358563
time: 5.36 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2227494, upper bound: 193.2228055
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2227494, upper bound: 193.2228055
time: 5.34 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2210914, upper bound: 193.2210914
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2210914, upper bound: 193.2210914
time: 5.05 seconds

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
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 246

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2308986, upper bound: 193.2308775
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2308775, upper bound: 193.2308986
time: 5.45 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 11.78 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.78
Output dim: 2, lower bound: -193.2359509, upper bound: 193.2359925
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.78
Output dim: 2, lower bound: -193.2359509, upper bound: 193.2359925
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 11.78
Output dim: 2, lower bound: -193.1399371, upper bound: 193.1399375
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 11.78
Output dim: 2, lower bound: -193.1399371, upper bound: 193.1399375
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.78
Output dim: 2, lower bound: -193.2074982, upper bound: 193.2074982
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.78
Output dim: 2, lower bound: -193.2074982, upper bound: 193.2074982
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.78
Output dim: 2, lower bound: -193.1535522, upper bound: 193.1535522
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.78
Output dim: 2, lower bound: -193.1535522, upper bound: 193.1535522
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.78
Output dim: 2, lower bound: -193.2359086, upper bound: 193.2358605
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.78
Output dim: 2, lower bound: -193.2359097, upper bound: 193.2358563
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.78
Output dim: 2, lower bound: -193.2227494, upper bound: 193.2228055
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.78
Output dim: 2, lower bound: -193.2227494, upper bound: 193.2228055
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.78
Output dim: 2, lower bound: -193.2210914, upper bound: 193.2210914
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.78
Output dim: 2, lower bound: -193.2210914, upper bound: 193.2210914
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.78
Output dim: 2, lower bound: -193.2308986, upper bound: 193.2308775
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.78
Output dim: 2, lower bound: -193.2308775, upper bound: 193.2308986

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1690540, upper bound: 193.1690668
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1690540, upper bound: 193.1690668
time: 5.26 seconds

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2337867, upper bound: 193.2338197
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2337867, upper bound: 193.2338197
time: 5.36 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2074642, upper bound: 193.2074676
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2074676, upper bound: 193.2074642
time: 5.53 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1531693, upper bound: 193.1531693
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1531693, upper bound: 193.1531693
time: 4.45 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1234990, upper bound: 193.1234986
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1234990, upper bound: 193.1234986
time: 4.49 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1535522, upper bound: 193.1535519
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1535519, upper bound: 193.1535522
time: 5.76 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2337594, upper bound: 193.2337350
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2337594, upper bound: 193.2337350
time: 6.11 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2084048, upper bound: 193.2083531
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2084048, upper bound: 193.2083531
time: 5.06 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2227033, upper bound: 193.2228055
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2227494, upper bound: 193.2227486
time: 5.51 seconds

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
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1724789, upper bound: 193.1724810
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1724789, upper bound: 193.1724810
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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 158

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1809343, upper bound: 193.1809343
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1809343, upper bound: 193.1809343
time: 4.81 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2101615, upper bound: 193.2101684
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2101684, upper bound: 193.2101615
time: 5.55 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2308986, upper bound: 193.2308775
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2308986, upper bound: 193.2308775
time: 5.62 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1375215, upper bound: 193.1374953
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1375215, upper bound: 193.1374953
time: 4.56 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 12.57 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.1690540, upper bound: 193.1690668
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.1690540, upper bound: 193.1690668
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.2337867, upper bound: 193.2338197
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.2337867, upper bound: 193.2338197
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.2074642, upper bound: 193.2074676
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.2074676, upper bound: 193.2074642
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.1531693, upper bound: 193.1531693
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.1531693, upper bound: 193.1531693
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.1234990, upper bound: 193.1234986
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.1234990, upper bound: 193.1234986
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.1535522, upper bound: 193.1535519
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.1535519, upper bound: 193.1535522
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.2337594, upper bound: 193.2337350
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.2337594, upper bound: 193.2337350
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.2084048, upper bound: 193.2083531
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.2084048, upper bound: 193.2083531
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.2227033, upper bound: 193.2228055
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.2227494, upper bound: 193.2227486
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.1724789, upper bound: 193.1724810
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.1724789, upper bound: 193.1724810
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.1809343, upper bound: 193.1809343
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.1809343, upper bound: 193.1809343
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.2101615, upper bound: 193.2101684
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.2101684, upper bound: 193.2101615
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.2308986, upper bound: 193.2308775
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.2308986, upper bound: 193.2308775
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.1375215, upper bound: 193.1374953
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.57
Output dim: 2, lower bound: -193.1375215, upper bound: 193.1374953

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 158

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688100, upper bound: 193.1688178
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1688100, upper bound: 193.1688178
time: 5.51 seconds

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1690522, upper bound: 193.1690668
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1690540, upper bound: 193.1690639
time: 5.09 seconds

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

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2337867, upper bound: 193.2338197
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2337867, upper bound: 193.2338197
time: 5.38 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2236252, upper bound: 193.2236960
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2236252, upper bound: 193.2236960
time: 5.46 seconds

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2040667, upper bound: 193.2040657
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2040667, upper bound: 193.2040657
time: 6.15 seconds

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

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2074676, upper bound: 193.2074566
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2074590, upper bound: 193.2074642
time: 5.64 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1531693, upper bound: 193.1531686
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1531686, upper bound: 193.1531693
time: 5.20 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1531693, upper bound: 193.1531641
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1531641, upper bound: 193.1531693
time: 5.36 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 158

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1466025, upper bound: 193.1465865
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1466025, upper bound: 193.1465865
time: 5.19 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 246

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1535240, upper bound: 193.1535522
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1535519, upper bound: 193.1535294
time: 5.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1661189, upper bound: 193.1661158
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1661189, upper bound: 193.1661158
time: 5.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2053061, upper bound: 193.2052975
time: 7.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2053061, upper bound: 193.2052975
time: 6.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 246

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1849087, upper bound: 193.1849031
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1849087, upper bound: 193.1849031
time: 5.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2014016, upper bound: 193.2013673
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2014016, upper bound: 193.2013673
time: 5.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2226923, upper bound: 193.2228055
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2227033, upper bound: 193.2227923
time: 5.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2217797, upper bound: 193.2218145
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2217770, upper bound: 193.2218360
time: 6.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 246

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=194.859130859375
rel_dist={2: [-193.2889031662745, 193.28890316692934]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 246

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1487995, upper bound: 193.1487995
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1487995, upper bound: 193.1487995
time: 5.49 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.99 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 10.99
Output dim: 2, lower bound: -193.1487995, upper bound: 193.1487995
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 10.99
Output dim: 2, lower bound: -193.1487995, upper bound: 193.1487995
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=194.859130859375
rel_dist={2: [-193.2888173876961, 193.28881738769616]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2839363, upper bound: 193.2839363
time: 6.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2839363, upper bound: 193.2839363
time: 6.20 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.93 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.93
Output dim: 2, lower bound: -193.2839363, upper bound: 193.2839363
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.93
Output dim: 2, lower bound: -193.2839363, upper bound: 193.2839363

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2839354, upper bound: 193.2839363
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2839363, upper bound: 193.2839354
time: 6.12 seconds

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
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2827476, upper bound: 193.2827476
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2827476, upper bound: 193.2827476
time: 6.40 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.23 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.23
Output dim: 2, lower bound: -193.2839354, upper bound: 193.2839363
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.23
Output dim: 2, lower bound: -193.2839363, upper bound: 193.2839354
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.23
Output dim: 2, lower bound: -193.2827476, upper bound: 193.2827476
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.23
Output dim: 2, lower bound: -193.2827476, upper bound: 193.2827476

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2657406, upper bound: 193.2657667
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2657406, upper bound: 193.2657667
time: 5.49 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2695647, upper bound: 193.2695477
time: 6.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2695647, upper bound: 193.2695477
time: 5.79 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2611096, upper bound: 193.2611096
time: 6.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2611096, upper bound: 193.2611096
time: 6.61 seconds

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
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2816077, upper bound: 193.2816077
time: 7.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2816077, upper bound: 193.2816077
time: 7.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.76 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.76
Output dim: 2, lower bound: -193.2657406, upper bound: 193.2657667
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.76
Output dim: 2, lower bound: -193.2657406, upper bound: 193.2657667
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.76
Output dim: 2, lower bound: -193.2695647, upper bound: 193.2695477
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.76
Output dim: 2, lower bound: -193.2695647, upper bound: 193.2695477
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.76
Output dim: 2, lower bound: -193.2611096, upper bound: 193.2611096
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.76
Output dim: 2, lower bound: -193.2611096, upper bound: 193.2611096
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.76
Output dim: 2, lower bound: -193.2816077, upper bound: 193.2816077
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.76
Output dim: 2, lower bound: -193.2816077, upper bound: 193.2816077

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1362844, upper bound: 193.1362781
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1362844, upper bound: 193.1362781
time: 5.23 seconds

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

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2574256, upper bound: 193.2574458
time: 5.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2574260, upper bound: 193.2574449
time: 4.34 seconds

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2687887, upper bound: 193.2687672
time: 8.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2687821, upper bound: 193.2687782
time: 7.63 seconds

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2695325, upper bound: 193.2695477
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2695647, upper bound: 193.2695142
time: 6.00 seconds

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
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2440868, upper bound: 193.2440868
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2440868, upper bound: 193.2440868
time: 6.30 seconds

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
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2611043, upper bound: 193.2611096
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2611096, upper bound: 193.2611043
time: 6.02 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1514215, upper bound: 193.1514215
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1514215, upper bound: 193.1514215
time: 4.75 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2551387, upper bound: 193.2551424
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2551387, upper bound: 193.2551424
time: 5.49 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 11.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 11.95
Output dim: 2, lower bound: -193.1362844, upper bound: 193.1362781
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 11.95
Output dim: 2, lower bound: -193.1362844, upper bound: 193.1362781
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.95
Output dim: 2, lower bound: -193.2574256, upper bound: 193.2574458
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.95
Output dim: 2, lower bound: -193.2574260, upper bound: 193.2574449
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.95
Output dim: 2, lower bound: -193.2687887, upper bound: 193.2687672
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.95
Output dim: 2, lower bound: -193.2687821, upper bound: 193.2687782
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.95
Output dim: 2, lower bound: -193.2695325, upper bound: 193.2695477
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.95
Output dim: 2, lower bound: -193.2695647, upper bound: 193.2695142
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.95
Output dim: 2, lower bound: -193.2440868, upper bound: 193.2440868
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.95
Output dim: 2, lower bound: -193.2440868, upper bound: 193.2440868
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.95
Output dim: 2, lower bound: -193.2611043, upper bound: 193.2611096
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.95
Output dim: 2, lower bound: -193.2611096, upper bound: 193.2611043
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.95
Output dim: 2, lower bound: -193.1514215, upper bound: 193.1514215
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.95
Output dim: 2, lower bound: -193.1514215, upper bound: 193.1514215
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.95
Output dim: 2, lower bound: -193.2551387, upper bound: 193.2551424
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.95
Output dim: 2, lower bound: -193.2551387, upper bound: 193.2551424

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

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2545682, upper bound: 193.2545990
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2545682, upper bound: 193.2545990
time: 6.37 seconds

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
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2533244, upper bound: 193.2533644
time: 6.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2533244, upper bound: 193.2533644
time: 6.50 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2687878, upper bound: 193.2687672
time: 5.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2687887, upper bound: 193.2687671
time: 6.94 seconds

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

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 246

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2687821, upper bound: 193.2687782
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2687821, upper bound: 193.2687782
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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1562674, upper bound: 193.1562674
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1562674, upper bound: 193.1562674
time: 5.84 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2557813, upper bound: 193.2557693
time: 7.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2557813, upper bound: 193.2557693
time: 7.23 seconds

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2440618, upper bound: 193.2440868
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2440868, upper bound: 193.2440618
time: 5.56 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1911360, upper bound: 193.1911360
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1911360, upper bound: 193.1911360
time: 5.68 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2376797, upper bound: 193.2376801
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2376797, upper bound: 193.2376801
time: 4.62 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2500050, upper bound: 193.2499976
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2500050, upper bound: 193.2499976
time: 6.43 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1368810, upper bound: 193.1368816
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1368810, upper bound: 193.1368816
time: 4.27 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 158

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1514215, upper bound: 193.1514214
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1514214, upper bound: 193.1514215
time: 4.99 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2551387, upper bound: 193.2551387
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2551340, upper bound: 193.2551424
time: 7.75 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2353757, upper bound: 193.2353829
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2353757, upper bound: 193.2353829
time: 6.10 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.48 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2545682, upper bound: 193.2545990
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2545682, upper bound: 193.2545990
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2533244, upper bound: 193.2533644
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2533244, upper bound: 193.2533644
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2687878, upper bound: 193.2687672
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2687887, upper bound: 193.2687671
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2687821, upper bound: 193.2687782
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2687821, upper bound: 193.2687782
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.1562674, upper bound: 193.1562674
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.1562674, upper bound: 193.1562674
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2557813, upper bound: 193.2557693
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2557813, upper bound: 193.2557693
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2440618, upper bound: 193.2440868
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2440868, upper bound: 193.2440618
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.1911360, upper bound: 193.1911360
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.1911360, upper bound: 193.1911360
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2376797, upper bound: 193.2376801
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2376797, upper bound: 193.2376801
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2500050, upper bound: 193.2499976
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2500050, upper bound: 193.2499976
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.1368810, upper bound: 193.1368816
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.1368810, upper bound: 193.1368816
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.1514215, upper bound: 193.1514214
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.1514214, upper bound: 193.1514215
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2551387, upper bound: 193.2551387
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2551340, upper bound: 193.2551424
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2353757, upper bound: 193.2353829
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.48
Output dim: 2, lower bound: -193.2353757, upper bound: 193.2353829

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2545145, upper bound: 193.2545990
time: 5.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2545682, upper bound: 193.2545291
time: 5.51 seconds

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
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2532349, upper bound: 193.2532440
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2532349, upper bound: 193.2532440
time: 5.59 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2533244, upper bound: 193.2533550
time: 5.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2533197, upper bound: 193.2533644
time: 5.13 seconds

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
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2533244, upper bound: 193.2533550
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2533197, upper bound: 193.2533644
time: 6.17 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2664541, upper bound: 193.2664191
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2664541, upper bound: 193.2664191
time: 7.06 seconds

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2512816, upper bound: 193.2512753
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2512816, upper bound: 193.2512753
time: 4.96 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1978873, upper bound: 193.1978875
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1978873, upper bound: 193.1978875
time: 5.79 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2358930, upper bound: 193.2359040
time: 6.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2358930, upper bound: 193.2359040
time: 6.58 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1528768, upper bound: 193.1528738
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1528768, upper bound: 193.1528738
time: 4.84 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1511334, upper bound: 193.1511334
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1511334, upper bound: 193.1511334
time: 5.00 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2532002, upper bound: 193.2531666
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2532002, upper bound: 193.2531666
time: 7.16 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2527166, upper bound: 193.2527113
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2527166, upper bound: 193.2527113
time: 5.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 246
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2440618, upper bound: 193.2440797
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2440574, upper bound: 193.2440868
time: 6.15 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 13.53 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2545145, upper bound: 193.2545990
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2545682, upper bound: 193.2545291
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2532349, upper bound: 193.2532440
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2532349, upper bound: 193.2532440
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2533244, upper bound: 193.2533550
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2533197, upper bound: 193.2533644
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2533244, upper bound: 193.2533550
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2533197, upper bound: 193.2533644
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2664541, upper bound: 193.2664191
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2664541, upper bound: 193.2664191
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2512816, upper bound: 193.2512753
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2512816, upper bound: 193.2512753
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.1978873, upper bound: 193.1978875
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.1978873, upper bound: 193.1978875
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2358930, upper bound: 193.2359040
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2358930, upper bound: 193.2359040
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.1528768, upper bound: 193.1528738
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.1528768, upper bound: 193.1528738
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.1511334, upper bound: 193.1511334
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.1511334, upper bound: 193.1511334
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2532002, upper bound: 193.2531666
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2532002, upper bound: 193.2531666
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2527166, upper bound: 193.2527113
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2527166, upper bound: 193.2527113
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2440618, upper bound: 193.2440797
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.53
Output dim: 2, lower bound: -193.2440574, upper bound: 193.2440868
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.2440868, upper bound: 193.2440618
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1911360, upper bound: 193.1911360
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1911360, upper bound: 193.1911360
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.2376797, upper bound: 193.2376801
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.2376797, upper bound: 193.2376801
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.2500050, upper bound: 193.2499976
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.2500050, upper bound: 193.2499976
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1514215, upper bound: 193.1514214
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.1514214, upper bound: 193.1514215
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.2551387, upper bound: 193.2551387
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.2551340, upper bound: 193.2551424
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.2353757, upper bound: 193.2353829
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 2, lower bound: -193.2353757, upper bound: 193.2353829
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=194.859130859375
rel_dist={2: [-193.2888482728832, 193.28884827288323]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1227.85 seconds
