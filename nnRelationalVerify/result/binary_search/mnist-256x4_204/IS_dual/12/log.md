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
execution time: IAR + LP analysis = 1.30 + 9.05 = 10.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -193.2890147, upper bound: 193.2890147


# Binary Search by BASE starts (time budget: 2689.65 seconds, max iter: 100)

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
Binary search time: 35.88 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2653.78 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2733919, upper bound: 193.2629149
time: 6.62 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2858051, upper bound: 193.2858051
time: 6.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.48 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.48
Output dim: 2, lower bound: -193.2733919, upper bound: 193.2629149
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.48
Output dim: 2, lower bound: -193.2858051, upper bound: 193.2858051

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -92.5710983, 73.7949295, -103.9151306, 82.6772232, -175.2483215, 177.7100525
1: -77.9647217, 65.4957733, -87.5886230, 73.4668121, -151.4315338, 153.0843658
2: -102.2964478, 66.4200058, -114.7588882, 74.5338287, -176.8302765, 181.1788788
3: -108.5205307, 57.7372856, -121.7540817, 64.7779617, -173.2984924, 179.4913330
4: -99.1955872, 76.3285904, -111.3217163, 85.6554260, -184.8510132, 187.6502838
5: -88.7699814, 69.1550369, -99.6948700, 77.7342148, -166.5041809, 168.8499146
6: -85.2733154, 82.2249146, -95.6775742, 92.2712479, -177.5445557, 177.9024811
7: -92.9629669, 78.0268402, -104.4919662, 87.6025009, -180.5654602, 182.5187378
8: -112.5315170, 77.0708542, -126.0867233, 86.3041534, -198.8356628, 203.1575775
9: -84.4136353, 83.4204712, -94.8750839, 93.6229782, -178.0366058, 178.2955475

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2587831, upper bound: 193.2587831
time: 6.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2587831, upper bound: 193.2629149
time: 5.93 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -102.6823044, 81.7061234, -106.9841232, 85.0895767, -187.7718658, 188.6902313
1: -86.5658798, 72.6101608, -90.1812515, 75.6211243, -162.1870117, 162.7914124
2: -113.4187164, 73.6618729, -118.1329117, 76.7262192, -190.1449280, 191.7947845
3: -120.3311996, 64.0238419, -125.3423386, 66.6745224, -187.0057220, 189.3661652
4: -109.9954300, 84.6645660, -114.6084290, 88.1751862, -198.1706085, 199.2729950
5: -98.5205078, 76.8254395, -102.6499710, 80.0399475, -178.5604401, 179.4754028
6: -94.5557632, 91.1921463, -98.4937134, 94.9893875, -189.5451508, 189.6858368
7: -103.2783661, 86.5848770, -107.6003113, 90.1861038, -193.4644775, 194.1851807
8: -124.6306076, 85.2908020, -129.7671661, 88.8023758, -213.4329834, 215.0579681
9: -93.7667542, 92.5312347, -97.6945114, 96.3775101, -190.1442413, 190.2257385

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2629149, upper bound: 193.2733919
time: 6.98 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2629149, upper bound: 193.2858051
time: 6.57 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 14.95 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.95
Output dim: 2, lower bound: -193.2587831, upper bound: 193.2587831
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.95
Output dim: 2, lower bound: -193.2587831, upper bound: 193.2629149
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.95
Output dim: 2, lower bound: -193.2629149, upper bound: 193.2733919
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.95
Output dim: 2, lower bound: -193.2629149, upper bound: 193.2858051

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -92.5710983, 73.7949295, -92.5710983, 73.7949295, -166.3659821, 166.3659821
1: -77.9647217, 65.4957733, -77.9647217, 65.4957733, -143.4604950, 143.4604950
2: -102.2964478, 66.4200058, -102.2964478, 66.4200058, -168.7164307, 168.7164154
3: -108.5205307, 57.7372856, -108.5205307, 57.7372856, -166.2577972, 166.2578125
4: -99.1955872, 76.3285904, -99.1955872, 76.3285904, -175.5241547, 175.5241547
5: -88.7699814, 69.1550369, -88.7699814, 69.1550369, -157.9250183, 157.9250183
6: -85.2733154, 82.2249146, -85.2733154, 82.2249146, -167.4981689, 167.4981689
7: -92.9629669, 78.0268402, -92.9629669, 78.0268402, -170.9897766, 170.9897766
8: -112.5315170, 77.0708542, -112.5315170, 77.0708542, -189.6023560, 189.6023560
9: -84.4136353, 83.4204712, -84.4136353, 83.4204712, -167.8341064, 167.8341064

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1907923, upper bound: 193.2092209
time: 6.03 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
time: 5.77 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -92.5710983, 73.7949295, -102.6823044, 81.7061234, -174.2771912, 176.4772186
1: -77.9647217, 65.4957733, -86.5658798, 72.6101608, -150.5748901, 152.0616455
2: -102.2964478, 66.4200058, -113.4187164, 73.6618729, -175.9582977, 179.8387146
3: -108.5205307, 57.7372856, -120.3311996, 64.0238419, -172.5443726, 178.0684814
4: -99.1955872, 76.3285904, -109.9954300, 84.6645660, -183.8601379, 186.3239899
5: -88.7699814, 69.1550369, -98.5205078, 76.8254395, -165.5954132, 167.6755371
6: -85.2733154, 82.2249146, -94.5557632, 91.1921463, -176.4654083, 176.7806396
7: -92.9629669, 78.0268402, -103.2783661, 86.5848770, -179.5478516, 181.3051453
8: -112.5315170, 77.0708542, -124.6306076, 85.2908020, -197.8223267, 201.7014465
9: -84.4136353, 83.4204712, -93.7667542, 92.5312347, -176.9448700, 177.1872253

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2092209, upper bound: 193.1968562
time: 5.74 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1957217
time: 5.00 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -102.6823044, 81.7061234, -92.5710983, 73.7949295, -176.4772186, 174.2771912
1: -86.5658798, 72.6101608, -77.9647217, 65.4957733, -152.0616455, 150.5748901
2: -113.4187164, 73.6618729, -102.2964478, 66.4200058, -179.8387146, 175.9582977
3: -120.3311996, 64.0238419, -108.5205307, 57.7372856, -178.0684814, 172.5443726
4: -109.9954300, 84.6645660, -99.1955872, 76.3285904, -186.3239899, 183.8601379
5: -98.5205078, 76.8254395, -88.7699814, 69.1550369, -167.6755371, 165.5954132
6: -94.5557632, 91.1921463, -85.2733154, 82.2249146, -176.7806396, 176.4654083
7: -103.2783661, 86.5848770, -92.9629669, 78.0268402, -181.3051453, 179.5478516
8: -124.6306076, 85.2908020, -112.5315170, 77.0708542, -201.7014465, 197.8223267
9: -93.7667542, 92.5312347, -84.4136353, 83.4204712, -177.1872253, 176.9448700

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1968562, upper bound: 193.2255166
time: 5.22 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1818086, upper bound: 193.2220970
time: 8.38 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -102.6823044, 81.7061234, -102.6823044, 81.7061234, -184.3884277, 184.3884277
1: -86.5658798, 72.6101608, -86.5658798, 72.6101608, -159.1760406, 159.1760406
2: -113.4187164, 73.6618729, -113.4187164, 73.6618729, -187.0805969, 187.0805969
3: -120.3311996, 64.0238419, -120.3311996, 64.0238419, -184.3550415, 184.3550415
4: -109.9954300, 84.6645660, -109.9954300, 84.6645660, -194.6599731, 194.6599731
5: -98.5205078, 76.8254395, -98.5205078, 76.8254395, -175.3459320, 175.3459320
6: -94.5557632, 91.1921463, -94.5557632, 91.1921463, -185.7478790, 185.7478790
7: -103.2783661, 86.5848770, -103.2783661, 86.5848770, -189.8632355, 189.8632355
8: -124.6306076, 85.2908020, -124.6306076, 85.2908020, -209.9214172, 209.9214172
9: -93.7667542, 92.5312347, -93.7667542, 92.5312347, -186.2979889, 186.2979889

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2408343, upper bound: 193.2714105
time: 6.86 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1957217, upper bound: 193.2220970
time: 6.43 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 14.72 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.72
Output dim: 2, lower bound: -193.1907923, upper bound: 193.2092209
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.72
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 14.72
Output dim: 2, lower bound: -193.2092209, upper bound: 193.1968562
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 14.72
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1957217
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.72
Output dim: 2, lower bound: -193.1968562, upper bound: 193.2255166
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.72
Output dim: 2, lower bound: -193.1818086, upper bound: 193.2220970
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 14.72
Output dim: 2, lower bound: -193.2408343, upper bound: 193.2714105
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 14.72
Output dim: 2, lower bound: -193.1957217, upper bound: 193.2220970

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -84.0246582, 67.0425644, -92.5710983, 73.7949295, -157.8195801, 159.6136017
1: -70.7055969, 59.4865685, -77.9647217, 65.4957733, -136.2013702, 137.4512939
2: -92.8490372, 60.2935104, -102.2964478, 66.4200058, -159.2690125, 162.5899506
3: -98.5477219, 52.4346275, -108.5205307, 57.7372856, -156.2849731, 160.9551544
4: -90.0032654, 69.2684631, -99.1955872, 76.3285904, -166.3318329, 168.4640503
5: -80.4947510, 62.7323189, -88.7699814, 69.1550369, -149.6497803, 151.5022888
6: -77.3808975, 74.5888672, -85.2733154, 82.2249146, -159.6057892, 159.8621826
7: -84.2931061, 70.7854767, -92.9629669, 78.0268402, -162.3198853, 163.7484436
8: -102.2184448, 69.9830246, -112.5315170, 77.0708542, -179.2893066, 182.5144958
9: -76.5220490, 75.6940460, -84.4136353, 83.4204712, -159.9425201, 160.1076813

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
time: 5.12 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
time: 4.91 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -82.6786575, 65.9964752, -89.3923035, 71.2825699, -153.9612274, 155.3887787
1: -69.5295792, 58.5330467, -75.2767792, 63.2630959, -132.7926788, 133.8097992
2: -91.4361420, 59.3242874, -98.7909622, 64.1460266, -155.5821686, 158.1152191
3: -97.0741730, 51.5299339, -104.8240204, 55.7650642, -152.8392181, 156.3539581
4: -88.5501251, 68.1753769, -95.7750702, 73.7152557, -162.2653656, 163.9504395
5: -79.1940460, 61.6374016, -85.6970978, 66.7677231, -145.9617615, 147.3345032
6: -76.1422424, 73.3440933, -82.3433762, 79.3905411, -155.5327759, 155.6874695
7: -82.9286499, 69.6382446, -89.7570114, 75.3447113, -158.2733612, 159.3952484
8: -100.6930008, 68.8089676, -108.7050476, 74.4300003, -175.1229858, 177.5139923
9: -75.1964569, 74.4152985, -81.4877777, 80.5487823, -155.7452240, 155.9030762

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
time: 4.72 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
time: 5.13 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -92.5710983, 73.7949295, -93.2157745, 74.2332306, -166.8043060, 167.0106812
1: -77.9647217, 65.4957733, -78.5291367, 65.9542618, -143.9189758, 144.0249023
2: -102.2964478, 66.4200058, -102.9597778, 66.8783493, -169.1747894, 169.3797760
3: -108.5205307, 57.7372856, -109.2818756, 58.1466141, -166.6671143, 167.0191498
4: -99.1955872, 76.3285904, -99.8193741, 76.8472061, -176.0427856, 176.1479492
5: -88.7699814, 69.1550369, -89.3522644, 69.7041397, -158.4740906, 158.5072937
6: -85.2733154, 82.2249146, -85.8189316, 82.7380447, -168.0113373, 168.0438538
7: -92.9629669, 78.0268402, -93.6724472, 78.5660324, -171.5289917, 171.6992340
8: -112.5315170, 77.0708542, -113.2140656, 77.4516983, -189.9831848, 190.2849121
9: -84.4136353, 83.4204712, -85.0225220, 83.9734421, -168.3870697, 168.4429932

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0994940, upper bound: 193.0869568
time: 5.74 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2249670, upper bound: 193.1964841
time: 5.83 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -89.3923035, 71.2825699, -95.7926102, 76.2786789, -165.6709900, 167.0751801
1: -75.2767792, 63.2630959, -80.6553192, 67.7581940, -143.0349731, 143.9184113
2: -98.7909622, 64.1460266, -105.8632736, 68.7125015, -167.5034637, 170.0093079
3: -104.8240204, 55.7650642, -112.3945236, 59.6664543, -164.4904785, 168.1595917
4: -95.7750702, 73.7152557, -102.5667572, 78.9613342, -174.7364044, 176.2820129
5: -85.6970978, 66.7677231, -91.8319016, 71.5602570, -157.2573547, 158.5996246
6: -82.3433762, 79.3905411, -88.1777344, 84.9729919, -167.3163300, 167.5682678
7: -89.7570114, 75.3447113, -96.2714691, 80.7301788, -170.4871826, 171.6161804
8: -108.7050476, 74.4300003, -116.4032059, 79.4741821, -188.1792145, 190.8332062
9: -81.4877777, 80.5487823, -87.3099365, 86.2237701, -167.7115173, 167.8587189

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1915669, upper bound: 193.1729294
time: 6.40 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2220970, upper bound: 193.1957217
time: 5.97 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -93.2157745, 74.2332306, -92.5710983, 73.7949295, -167.0106812, 166.8043060
1: -78.5291367, 65.9542618, -77.9647217, 65.4957733, -144.0249023, 143.9189758
2: -102.9597778, 66.8783493, -102.2964478, 66.4200058, -169.3797607, 169.1747894
3: -109.2818756, 58.1466141, -108.5205307, 57.7372856, -167.0191498, 166.6671143
4: -99.8193741, 76.8472061, -99.1955872, 76.3285904, -176.1479492, 176.0427856
5: -89.3522644, 69.7041397, -88.7699814, 69.1550369, -158.5072937, 158.4740906
6: -85.8189316, 82.7380447, -85.2733154, 82.2249146, -168.0438538, 168.0113373
7: -93.6724472, 78.5660324, -92.9629669, 78.0268402, -171.6992340, 171.5289917
8: -113.2140656, 77.4516983, -112.5315170, 77.0708542, -190.2849121, 189.9831848
9: -85.0225220, 83.9734421, -84.4136353, 83.4204712, -168.4429932, 168.3870697

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0869568, upper bound: 193.0994941
time: 5.75 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1964841, upper bound: 193.2249670
time: 6.86 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -95.7926102, 76.2786789, -89.3923035, 71.2825699, -167.0751801, 165.6709900
1: -80.6553192, 67.7581940, -75.2767792, 63.2630959, -143.9184113, 143.0349731
2: -105.8632736, 68.7125015, -98.7909622, 64.1460266, -170.0093079, 167.5034637
3: -112.3945236, 59.6664543, -104.8240204, 55.7650642, -168.1595917, 164.4904785
4: -102.5667572, 78.9613342, -95.7750702, 73.7152557, -176.2820129, 174.7364044
5: -91.8319016, 71.5602570, -85.6970978, 66.7677231, -158.5996246, 157.2573547
6: -88.1777344, 84.9729919, -82.3433762, 79.3905411, -167.5682678, 167.3163300
7: -96.2714691, 80.7301788, -89.7570114, 75.3447113, -171.6161804, 170.4871826
8: -116.4032059, 79.4741821, -108.7050476, 74.4300003, -190.8332062, 188.1792145
9: -87.3099365, 86.2237701, -81.4877777, 80.5487823, -167.8587189, 167.7115173

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1729294, upper bound: 193.1915669
time: 6.26 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1957217, upper bound: 193.2220970
time: 7.07 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -102.6823044, 81.7061234, -93.2157745, 74.2332306, -176.9155273, 174.9218903
1: -86.5658798, 72.6101608, -78.5291367, 65.9542618, -152.5201263, 151.1392822
2: -113.4187164, 73.6618729, -102.9597778, 66.8783493, -180.2970581, 176.6216431
3: -120.3311996, 64.0238419, -109.2818756, 58.1466141, -178.4777985, 173.3057098
4: -109.9954300, 84.6645660, -99.8193741, 76.8472061, -186.8426056, 184.4839478
5: -98.5205078, 76.8254395, -89.3522644, 69.7041397, -168.2246094, 166.1777039
6: -94.5557632, 91.1921463, -85.8189316, 82.7380447, -177.2938080, 177.0110779
7: -103.2783661, 86.5848770, -93.6724472, 78.5660324, -181.8443756, 180.2573242
8: -124.6306076, 85.2908020, -113.2140656, 77.4516983, -202.0822754, 198.5048676
9: -93.7667542, 92.5312347, -85.0225220, 83.9734421, -177.7401886, 177.5537567

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2530648, upper bound: 193.2564333
time: 5.90 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2457052, upper bound: 193.2384512
time: 5.76 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -99.7875290, 79.4152222, -95.7926102, 76.2786789, -176.0662079, 175.2078247
1: -84.1171341, 70.5770187, -80.6553192, 67.7581940, -151.8753204, 151.2323151
2: -110.2250214, 71.5913162, -105.8632736, 68.7125015, -178.9375305, 177.4545898
3: -116.9647598, 62.2275848, -112.3945236, 59.6664543, -176.6312103, 174.6221008
4: -106.8792038, 82.2835236, -102.5667572, 78.9613342, -185.8405457, 184.8502808
5: -95.7231903, 74.6541290, -91.8319016, 71.5602570, -167.2834167, 166.4860229
6: -91.8864288, 88.6109314, -88.1777344, 84.9729919, -176.8594208, 176.7886658
7: -100.3616486, 84.1448364, -96.2714691, 80.7301788, -181.0918274, 180.4163055
8: -121.1431503, 82.8822403, -116.4032059, 79.4741821, -200.6173096, 199.2854462
9: -91.1052475, 89.9167328, -87.3099365, 86.2237701, -177.3289795, 177.2266693

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2397313, upper bound: 193.2502395
time: 6.34 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2355005, upper bound: 193.2355077
time: 5.26 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 13.09 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.09
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.09
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.09
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.09
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
IS_A1_B2_B1_B1, status: Status.VERIFIED, split count: 4, time: 13.09
Output dim: 2, lower bound: -193.0994940, upper bound: 193.0869568
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 13.09
Output dim: 2, lower bound: -193.2249670, upper bound: 193.1964841
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 13.09
Output dim: 2, lower bound: -193.1915669, upper bound: 193.1729294
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 13.09
Output dim: 2, lower bound: -193.2220970, upper bound: 193.1957217
IS_A2_B1_A1_A1, status: Status.VERIFIED, split count: 4, time: 13.09
Output dim: 2, lower bound: -193.0869568, upper bound: 193.0994941
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 13.09
Output dim: 2, lower bound: -193.1964841, upper bound: 193.2249670
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 13.09
Output dim: 2, lower bound: -193.1729294, upper bound: 193.1915669
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 13.09
Output dim: 2, lower bound: -193.1957217, upper bound: 193.2220970
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 13.09
Output dim: 2, lower bound: -193.2530648, upper bound: 193.2564333
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 13.09
Output dim: 2, lower bound: -193.2457052, upper bound: 193.2384512
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 13.09
Output dim: 2, lower bound: -193.2397313, upper bound: 193.2502395
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 13.09
Output dim: 2, lower bound: -193.2355005, upper bound: 193.2355077

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -84.0246582, 67.0425644, -84.0246582, 67.0425644, -151.0672150, 151.0672150
1: -70.7055969, 59.4865685, -70.7055969, 59.4865685, -130.1921692, 130.1921692
2: -92.8490372, 60.2935104, -92.8490372, 60.2935104, -153.1425476, 153.1425476
3: -98.5477219, 52.4346275, -98.5477219, 52.4346275, -150.9823303, 150.9823303
4: -90.0032654, 69.2684631, -90.0032654, 69.2684631, -159.2717285, 159.2717285
5: -80.4947510, 62.7323189, -80.4947510, 62.7323189, -143.2270660, 143.2270660
6: -77.3808975, 74.5888672, -77.3808975, 74.5888672, -151.9697571, 151.9697571
7: -84.2931061, 70.7854767, -84.2931061, 70.7854767, -155.0785675, 155.0785675
8: -102.2184448, 69.9830246, -102.2184448, 69.9830246, -172.2014465, 172.2014465
9: -76.5220490, 75.6940460, -76.5220490, 75.6940460, -152.2160950, 152.2160950

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1566132, upper bound: 193.1655774
time: 6.25 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1907923, upper bound: 193.2092209
time: 6.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -84.0246582, 67.0425644, -82.6786575, 65.9964752, -150.0211334, 149.7211914
1: -70.7055969, 59.4865685, -69.5295792, 58.5330467, -129.2386322, 129.0161438
2: -92.8490372, 60.2935104, -91.4361420, 59.3242874, -152.1733093, 151.7296448
3: -98.5477219, 52.4346275, -97.0741730, 51.5299339, -150.0776215, 149.5088043
4: -90.0032654, 69.2684631, -88.5501251, 68.1753769, -158.1786499, 157.8185883
5: -80.4947510, 62.7323189, -79.1940460, 61.6374016, -142.1321564, 141.9263611
6: -77.3808975, 74.5888672, -76.1422424, 73.3440933, -150.7249908, 150.7311096
7: -84.2931061, 70.7854767, -82.9286499, 69.6382446, -153.9313354, 153.7141266
8: -102.2184448, 69.9830246, -100.6930008, 68.8089676, -171.0274048, 170.6760101
9: -76.5220490, 75.6940460, -75.1964569, 74.4152985, -150.9373474, 150.8904877

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1566132, upper bound: 193.1655774
time: 5.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1907923, upper bound: 193.2092209
time: 6.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -82.6786575, 65.9964752, -84.0246582, 67.0425644, -149.7211914, 150.0211334
1: -69.5295792, 58.5330467, -70.7055969, 59.4865685, -129.0161285, 129.2386322
2: -91.4361420, 59.3242874, -92.8490372, 60.2935104, -151.7296448, 152.1733093
3: -97.0741730, 51.5299339, -98.5477219, 52.4346275, -149.5088043, 150.0776215
4: -88.5501251, 68.1753769, -90.0032654, 69.2684631, -157.8185883, 158.1786499
5: -79.1940460, 61.6374016, -80.4947510, 62.7323189, -141.9263611, 142.1321564
6: -76.1422424, 73.3440933, -77.3808975, 74.5888672, -150.7311096, 150.7249908
7: -82.9286499, 69.6382446, -84.2931061, 70.7854767, -153.7141266, 153.9313354
8: -100.6930008, 68.8089676, -102.2184448, 69.9830246, -170.6760101, 171.0274048
9: -75.1964569, 74.4152985, -76.5220490, 75.6940460, -150.8904877, 150.9373474

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1044976, upper bound: 193.0772390
time: 5.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0633667, upper bound: 193.0633667
time: 5.20 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -82.6786575, 65.9964752, -82.6786575, 65.9964752, -148.6751099, 148.6751099
1: -69.5295792, 58.5330467, -69.5295792, 58.5330467, -128.0626068, 128.0626068
2: -91.4361420, 59.3242874, -91.4361420, 59.3242874, -150.7604370, 150.7604370
3: -97.0741730, 51.5299339, -97.0741730, 51.5299339, -148.6040955, 148.6040955
4: -88.5501251, 68.1753769, -88.5501251, 68.1753769, -156.7254944, 156.7254944
5: -79.1940460, 61.6374016, -79.1940460, 61.6374016, -140.8314514, 140.8314514
6: -76.1422424, 73.3440933, -76.1422424, 73.3440933, -149.4863281, 149.4863281
7: -82.9286499, 69.6382446, -82.9286499, 69.6382446, -152.5668945, 152.5668945
8: -100.6930008, 68.8089676, -100.6930008, 68.8089676, -169.5019684, 169.5019684
9: -75.1964569, 74.4152985, -75.1964569, 74.4152985, -149.6117401, 149.6117401

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0772390, upper bound: 193.1044976
time: 5.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0633667, upper bound: 193.0633667
time: 4.41 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -92.5710983, 73.7949295, -92.0026627, 73.2837906, -165.8548279, 165.7975769
1: -77.9647217, 65.4957733, -77.5126953, 65.1054382, -143.0701599, 143.0084534
2: -102.2964478, 66.4200058, -101.6353760, 66.0205460, -168.3169861, 168.0553894
3: -108.5205307, 57.7372856, -107.8710022, 57.3831825, -165.9037170, 165.6082916
4: -99.1955872, 76.3285904, -98.5241852, 75.8619232, -175.0575104, 174.8527374
5: -88.7699814, 69.1550369, -88.1966858, 68.7977753, -157.5677338, 157.3517151
6: -85.2733154, 82.2249146, -84.7078018, 81.6716843, -166.9449615, 166.9327087
7: -92.9629669, 78.0268402, -92.4599609, 77.5569763, -170.5199280, 170.4867859
8: -112.5315170, 77.0708542, -111.7674484, 76.4509811, -188.9824982, 188.8382874
9: -84.4136353, 83.4204712, -83.9256668, 82.8935547, -167.3071899, 167.3461151

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2249670, upper bound: 193.1964841
time: 6.35 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2249670, upper bound: 193.1964841
time: 6.74 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -88.5189590, 70.5868301, -97.2910385, 77.4970245, -166.0159760, 167.8778687
1: -74.5482635, 62.6516380, -81.9980621, 68.8549118, -143.4031677, 144.6497040
2: -97.8278351, 63.5259285, -107.5442810, 69.8084335, -167.6362610, 171.0702057
3: -103.8004074, 55.2276154, -114.2147522, 60.5873566, -164.3877411, 169.4423676
4: -94.8351517, 73.0024567, -104.2453308, 80.2450943, -175.0802460, 177.2477722
5: -84.8601837, 66.1178055, -93.3591690, 72.7184906, -157.5786743, 159.4769440
6: -81.5404510, 78.6191940, -89.5923386, 86.3498077, -167.8902588, 168.2115173
7: -88.8816223, 74.6134262, -97.8270340, 82.0280228, -170.9096375, 172.4404602
8: -107.6518097, 73.7134094, -118.2558060, 80.7415543, -188.3933716, 191.9691925
9: -80.6909637, 79.7649155, -88.6840668, 87.5876617, -168.2786255, 168.4489594

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1500582, upper bound: 193.1245764
time: 5.95 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1219426, upper bound: 193.1030909
time: 6.76 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -89.3923035, 71.2825699, -94.9061584, 75.5720291, -164.9643250, 166.1887207
1: -75.2767792, 63.2630959, -79.9165268, 67.1383896, -142.4151611, 143.1796265
2: -98.7909622, 64.1460266, -104.8871994, 68.0826569, -166.8735809, 169.0332336
3: -104.8240204, 55.7650642, -111.3590851, 59.1232262, -163.9472504, 167.1241302
4: -95.7750702, 73.7152557, -101.6137161, 78.2383041, -174.0133667, 175.3289490
5: -85.6970978, 66.7677231, -90.9843369, 70.8998642, -156.5969543, 157.7520599
6: -82.3433762, 79.3905411, -87.3649521, 84.1897278, -166.5330658, 166.7554932
7: -89.7570114, 75.3447113, -95.3841476, 79.9883347, -169.7453461, 170.7288513
8: -108.7050476, 74.4300003, -115.3323212, 78.7468643, -187.4518890, 189.7622986
9: -81.4877777, 80.5487823, -86.5003128, 85.4274750, -166.9152527, 167.0491028

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 81

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2220557, upper bound: 193.1957217
time: 7.52 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2220557, upper bound: 193.1956838
time: 7.16 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -92.0026627, 73.2837906, -92.5710983, 73.7949295, -165.7975769, 165.8548279
1: -77.5126953, 65.1054382, -77.9647217, 65.4957733, -143.0084534, 143.0701599
2: -101.6353760, 66.0205460, -102.2964478, 66.4200058, -168.0553894, 168.3169861
3: -107.8710022, 57.3831825, -108.5205307, 57.7372856, -165.6082916, 165.9037170
4: -98.5241852, 75.8619232, -99.1955872, 76.3285904, -174.8527374, 175.0575104
5: -88.1966858, 68.7977753, -88.7699814, 69.1550369, -157.3517151, 157.5677338
6: -84.7078018, 81.6716843, -85.2733154, 82.2249146, -166.9327087, 166.9449615
7: -92.4599609, 77.5569763, -92.9629669, 78.0268402, -170.4867859, 170.5199280
8: -111.7674484, 76.4509811, -112.5315170, 77.0708542, -188.8382874, 188.9824982
9: -83.9256668, 82.8935547, -84.4136353, 83.4204712, -167.3461151, 167.3071899

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1964841, upper bound: 193.2249670
time: 6.49 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1964841, upper bound: 193.2249670
time: 7.23 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -97.2910385, 77.4970245, -88.5189590, 70.5868301, -167.8778687, 166.0159760
1: -81.9980621, 68.8549118, -74.5482635, 62.6516380, -144.6497040, 143.4031677
2: -107.5442810, 69.8084335, -97.8278351, 63.5259285, -171.0702057, 167.6362610
3: -114.2147522, 60.5873566, -103.8004074, 55.2276154, -169.4423676, 164.3877411
4: -104.2453308, 80.2450943, -94.8351517, 73.0024567, -177.2477722, 175.0802460
5: -93.3591690, 72.7184906, -84.8601837, 66.1178055, -159.4769440, 157.5786743
6: -89.5923386, 86.3498077, -81.5404510, 78.6191940, -168.2115173, 167.8902588
7: -97.8270340, 82.0280228, -88.8816223, 74.6134262, -172.4404602, 170.9096375
8: -118.2558060, 80.7415543, -107.6518097, 73.7134094, -191.9691925, 188.3933716
9: -88.6840668, 87.5876617, -80.6909637, 79.7649155, -168.4489594, 168.2786255

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1245758, upper bound: 193.1500582
time: 7.50 seconds

## Relational analysis of IS_A2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1030909, upper bound: 193.1219426
time: 5.64 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -94.9061584, 75.5720291, -89.3923035, 71.2825699, -166.1887207, 164.9643250
1: -79.9165268, 67.1383896, -75.2767792, 63.2630959, -143.1796265, 142.4151611
2: -104.8871994, 68.0826569, -98.7909622, 64.1460266, -169.0332336, 166.8735809
3: -111.3590851, 59.1232262, -104.8240204, 55.7650642, -167.1241302, 163.9472504
4: -101.6137161, 78.2383041, -95.7750702, 73.7152557, -175.3289490, 174.0133667
5: -90.9843369, 70.8998642, -85.6970978, 66.7677231, -157.7520599, 156.5969391
6: -87.3649521, 84.1897278, -82.3433762, 79.3905411, -166.7554932, 166.5330658
7: -95.3841476, 79.9883347, -89.7570114, 75.3447113, -170.7288513, 169.7453461
8: -115.3323212, 78.7468643, -108.7050476, 74.4300003, -189.7622986, 187.4518890
9: -86.5003128, 85.4274750, -81.4877777, 80.5487823, -167.0491028, 166.9152527

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 81

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1957217, upper bound: 193.2220970
time: 6.53 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1957217, upper bound: 193.2220970
time: 5.12 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -98.7086792, 78.5337143, -93.2157745, 74.2332306, -172.9419098, 171.7494812
1: -83.1943436, 69.7972946, -78.5291367, 65.9542618, -149.1485443, 148.3264313
2: -108.9949417, 70.8004913, -102.9597778, 66.8783493, -175.8732910, 173.7602692
3: -115.6383896, 61.5559425, -109.2818756, 58.1466141, -173.7849579, 170.8377991
4: -105.7363586, 81.3689728, -99.8193741, 76.8472061, -182.5835571, 181.1883392
5: -94.6732712, 73.8499527, -89.3522644, 69.7041397, -164.3773956, 163.2022095
6: -90.8665009, 87.6581955, -85.8189316, 82.7380447, -173.6045380, 173.4771271
7: -99.2598419, 83.2156448, -93.6724472, 78.5660324, -177.8258667, 176.8880920
8: -119.7823105, 82.0000534, -113.2140656, 77.4516983, -197.2339935, 195.2141113
9: -90.0953979, 88.9330597, -85.0225220, 83.9734421, -174.0688477, 173.9555664

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2530648, upper bound: 193.2564333
time: 6.94 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2530648, upper bound: 193.2564333
time: 6.90 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -97.0437317, 77.1408310, -91.4252701, 72.8021011, -169.8458252, 168.5660858
1: -81.7117004, 68.5821457, -77.0132523, 64.6890564, -146.4007416, 145.5953979
2: -107.0785522, 69.5026932, -100.9686203, 65.5888672, -172.6674194, 170.4712830
3: -113.6204376, 60.4641380, -107.1717834, 57.0369415, -170.6573792, 167.6359100
4: -103.9695663, 79.9399185, -97.8968887, 75.3649597, -179.3345337, 177.8367920
5: -93.0068436, 72.5592957, -87.6205902, 68.3640060, -161.3707886, 160.1798859
6: -89.2558670, 86.1473770, -84.1585999, 81.1468201, -170.4026489, 170.3059692
7: -97.5234528, 81.7417908, -91.8643341, 77.0505981, -174.5740204, 173.6061096
8: -117.6908569, 80.5541992, -111.0321121, 75.9682236, -193.6590729, 191.5863037
9: -88.5022354, 87.3560181, -83.3735275, 82.3521347, -170.8543701, 170.7295532

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 210

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2457052, upper bound: 193.2384512
time: 6.22 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2457052, upper bound: 193.2384512
time: 6.41 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -95.8176117, 76.2456284, -95.7926102, 76.2786789, -172.0962677, 172.0382385
1: -80.7486877, 67.7672653, -80.6553192, 67.7581940, -148.5068512, 148.4225464
2: -105.8047562, 68.7328339, -105.8632736, 68.7125015, -174.5172577, 174.5960999
3: -112.2766647, 59.7620583, -112.3945236, 59.6664543, -171.9430847, 172.1565704
4: -102.6236801, 78.9912949, -102.5667572, 78.9613342, -181.5850220, 181.5580292
5: -91.8795547, 71.6815872, -91.8319016, 71.5602570, -163.4398193, 163.5134735
6: -88.2005463, 85.0798721, -88.1777344, 84.9729919, -173.1735077, 173.2575989
7: -96.3466339, 80.7785645, -96.2714691, 80.7301788, -177.0768127, 177.0500336
8: -116.2993622, 79.5945663, -116.4032059, 79.4741821, -195.7735443, 195.9977722
9: -87.4374313, 86.3219376, -87.3099365, 86.2237701, -173.6611633, 173.6318665

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1814093, upper bound: 193.1793986
time: 6.74 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2393302, upper bound: 193.2500688
time: 5.99 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -94.0240784, 74.7479553, -94.0169601, 74.8589783, -168.8830109, 168.7649231
1: -79.1538696, 66.4623337, -79.1493454, 66.5021744, -145.6560364, 145.6116791
2: -103.7462616, 67.3417130, -103.8868332, 67.4330139, -171.1792450, 171.2285461
3: -110.1053696, 58.5939751, -110.2998734, 58.5651550, -168.6705322, 168.8938446
4: -100.7210846, 77.4559631, -100.6594696, 77.4902039, -178.2112579, 178.1154327
5: -90.0848236, 70.2981033, -90.1132736, 70.2308960, -160.3157043, 160.4113464
6: -86.4734039, 83.4545135, -86.5302658, 83.3939056, -169.8673096, 169.9847565
7: -94.4794235, 79.1936722, -94.4766083, 79.2256699, -173.7050934, 173.6702728
8: -114.0531769, 78.0387192, -114.2374649, 78.0027771, -192.0559235, 192.2761688
9: -85.7226486, 84.6266174, -85.6732483, 84.6147766, -170.3374329, 170.2998657

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 81

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2355005, upper bound: 193.2355077
time: 6.43 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2355005, upper bound: 193.2355077
time: 6.31 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 18.91 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.1566132, upper bound: 193.1655774
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.1907923, upper bound: 193.2092209
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.1566132, upper bound: 193.1655774
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.1907923, upper bound: 193.2092209
IS_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.1044976, upper bound: 193.0772390
IS_A1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.0633667, upper bound: 193.0633667
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.0772390, upper bound: 193.1044976
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.0633667, upper bound: 193.0633667
IS_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.2249670, upper bound: 193.1964841
IS_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.2249670, upper bound: 193.1964841
IS_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.1500582, upper bound: 193.1245764
IS_A1_B2_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.1219426, upper bound: 193.1030909
IS_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.2220557, upper bound: 193.1957217
IS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.2220557, upper bound: 193.1956838
IS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.1964841, upper bound: 193.2249670
IS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.1964841, upper bound: 193.2249670
IS_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.1245758, upper bound: 193.1500582
IS_A2_B1_A2_A1_A2, status: Status.VERIFIED, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.1030909, upper bound: 193.1219426
IS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.1957217, upper bound: 193.2220970
IS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.1957217, upper bound: 193.2220970
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.2530648, upper bound: 193.2564333
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.2530648, upper bound: 193.2564333
IS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.2457052, upper bound: 193.2384512
IS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.2457052, upper bound: 193.2384512
IS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.1814093, upper bound: 193.1793986
IS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.2393302, upper bound: 193.2500688
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.2355005, upper bound: 193.2355077
IS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 18.91
Output dim: 2, lower bound: -193.2355005, upper bound: 193.2355077

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -85.1485062, 67.9637604, -83.1534500, 66.3478317, -151.4963074, 151.1172028
1: -71.7405014, 60.3241882, -69.9786835, 58.8759727, -130.6164703, 130.3028412
2: -94.1061020, 61.1023521, -91.8876343, 59.6745110, -153.7806091, 152.9899445
3: -99.9318695, 53.1298332, -97.5260620, 51.8986206, -151.8304901, 150.6558990
4: -91.2722778, 70.2413864, -89.0661697, 68.5576859, -159.8299561, 159.3075409
5: -81.6668320, 63.6201477, -79.6599503, 62.0834122, -143.7502441, 143.2800751
6: -78.4491959, 75.6353989, -76.5799103, 73.8192291, -152.2683868, 152.2152863
7: -85.4610596, 71.7649841, -83.4193344, 70.0553131, -155.5163727, 155.1843262
8: -103.6087112, 70.9453812, -101.1669159, 69.2681046, -172.8768005, 172.1122894
9: -77.5579834, 76.7168274, -75.7264786, 74.9119720, -152.4699402, 152.4432983

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2307478, upper bound: 193.2239884
time: 6.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2292245, upper bound: 193.2210060
time: 6.11 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -83.1461487, 66.3432465, -84.0246582, 67.0425644, -150.1886902, 150.3679047
1: -69.9739304, 58.8728027, -70.7055969, 59.4865685, -129.4604797, 129.5783997
2: -91.8823700, 59.6696014, -92.8490372, 60.2935104, -152.1758728, 152.5186462
3: -97.5219345, 51.8970795, -98.5477219, 52.4346275, -149.9565582, 150.4447784
4: -89.0592728, 68.5525665, -90.0032654, 69.2684631, -158.3277283, 158.5558167
5: -79.6558075, 62.0785484, -80.4947510, 62.7323189, -142.3881226, 142.5733032
6: -76.5756836, 73.8128891, -77.3808975, 74.5888672, -151.1645508, 151.1937866
7: -83.4139404, 70.0508347, -84.2931061, 70.7854767, -154.1994171, 154.3439331
8: -101.1563110, 69.2624512, -102.2184448, 69.9830246, -171.1393280, 171.4808807
9: -75.7189713, 74.9055634, -76.5220490, 75.6940460, -151.4130249, 151.4276123

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1660122, upper bound: 193.1889080
time: 6.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2472552, upper bound: 193.2472552
time: 6.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -85.1485062, 67.9637604, -81.8055038, 65.2996826, -150.4481659, 149.7692566
1: -71.7405014, 60.3241882, -68.8006134, 57.9205780, -129.6610718, 129.1247864
2: -94.1061020, 61.1023521, -90.4729462, 58.7039795, -152.8100739, 151.5752869
3: -99.9318695, 53.1298332, -96.0496063, 50.9926071, -150.9244690, 149.1793976
4: -91.2722778, 70.2413864, -87.6113205, 67.4626465, -158.7349243, 157.8526764
5: -81.6668320, 63.6201477, -78.3567047, 60.9863052, -142.6531219, 141.9768524
6: -78.4491959, 75.6353989, -75.3395462, 72.5722809, -151.0214691, 150.9749451
7: -85.4610596, 71.7649841, -82.0530396, 68.9057083, -154.3667603, 153.8180237
8: -103.6087112, 70.9453812, -99.6390228, 68.0926437, -171.7013550, 170.5844116
9: -77.5579834, 76.7168274, -74.3985901, 73.6312790, -151.1892548, 151.1154175

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=194.859130859375
rel_dist={2: [-193.2889031662745, 193.28890316692934]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2673907, upper bound: 193.2609621
time: 9.04 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2857449, upper bound: 193.2857449
time: 7.22 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.42 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.42
Output dim: 2, lower bound: -193.2673907, upper bound: 193.2609621
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.42
Output dim: 2, lower bound: -193.2857449, upper bound: 193.2857449

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -92.5710983, 73.7949295, -99.3148727, 79.0612411, -171.6323090, 173.1098022
1: -77.9647217, 65.4957733, -83.7003021, 70.2376175, -148.2023315, 149.1960754
2: -102.2964478, 66.4200058, -109.7009506, 71.2486572, -173.5451050, 176.1209564
3: -108.5205307, 57.7372856, -116.3762589, 61.9350319, -170.4555664, 174.1135101
4: -99.1955872, 76.3285904, -106.3942719, 81.8774872, -181.0730591, 182.7228546
5: -88.7699814, 69.1550369, -95.2647018, 74.2769470, -163.0469055, 164.4197388
6: -85.2733154, 82.2249146, -91.4556808, 88.1958237, -173.4691010, 173.6805420
7: -92.9629669, 78.0268402, -99.8324280, 83.7285156, -176.6914825, 177.8592072
8: -112.5315170, 77.0708542, -120.5683365, 82.5587387, -195.0902252, 197.6391907
9: -84.4136353, 83.4204712, -90.6482620, 89.4937286, -173.9073639, 174.0686951

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2587072, upper bound: 193.2587072
time: 6.34 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2587072, upper bound: 193.2609621
time: 6.28 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -102.6823044, 81.7061234, -106.0542908, 84.3584595, -187.0407715, 187.7604065
1: -86.5658798, 72.6101608, -89.3999023, 74.9704971, -161.5363770, 162.0100250
2: -113.4187164, 73.6618729, -117.1142960, 76.0635986, -189.4823151, 190.7761536
3: -120.3311996, 64.0238419, -124.2591782, 66.1017609, -186.4329529, 188.2830048
4: -109.9954300, 84.6645660, -113.6116714, 87.4164810, -197.4118958, 198.2762146
5: -98.5205078, 76.8254395, -101.7572937, 79.3449707, -177.8654633, 178.5827332
6: -94.5557632, 91.1921463, -97.6425323, 94.1689301, -188.7246704, 188.8346252
7: -103.2783661, 86.5848770, -106.6662598, 89.4077225, -192.6860657, 193.2511292
8: -124.6306076, 85.2908020, -128.6567841, 88.0439911, -212.6745911, 213.9475861
9: -93.7667542, 92.5312347, -96.8455048, 95.5461578, -189.3128967, 189.3767395

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2609622, upper bound: 193.2673907
time: 7.54 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2609622, upper bound: 193.2857449
time: 6.86 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.09 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.09
Output dim: 2, lower bound: -193.2587072, upper bound: 193.2587072
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.09
Output dim: 2, lower bound: -193.2587072, upper bound: 193.2609621
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 16.09
Output dim: 2, lower bound: -193.2609622, upper bound: 193.2673907
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 16.09
Output dim: 2, lower bound: -193.2609622, upper bound: 193.2857449

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -92.5710983, 73.7949295, -92.5710983, 73.7949295, -166.3659821, 166.3659821
1: -77.9647217, 65.4957733, -77.9647217, 65.4957733, -143.4604950, 143.4604950
2: -102.2964478, 66.4200058, -102.2964478, 66.4200058, -168.7164307, 168.7164154
3: -108.5205307, 57.7372856, -108.5205307, 57.7372856, -166.2577972, 166.2578125
4: -99.1955872, 76.3285904, -99.1955872, 76.3285904, -175.5241547, 175.5241547
5: -88.7699814, 69.1550369, -88.7699814, 69.1550369, -157.9250183, 157.9250183
6: -85.2733154, 82.2249146, -85.2733154, 82.2249146, -167.4981689, 167.4981689
7: -92.9629669, 78.0268402, -92.9629669, 78.0268402, -170.9897766, 170.9897766
8: -112.5315170, 77.0708542, -112.5315170, 77.0708542, -189.6023560, 189.6023560
9: -84.4136353, 83.4204712, -84.4136353, 83.4204712, -167.8341064, 167.8341064

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1779398, upper bound: 193.1906249
time: 6.82 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2587072, upper bound: 193.2587072
time: 6.02 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -92.5710983, 73.7949295, -102.6648331, 81.6923599, -174.2634430, 176.4597168
1: -77.9647217, 65.4957733, -86.5521317, 72.5980682, -150.5627899, 152.0478973
2: -102.2964478, 66.4200058, -113.3997192, 73.6488419, -175.9452515, 179.8197174
3: -108.5205307, 57.7372856, -120.3111496, 64.0133591, -172.5338745, 178.0484161
4: -99.1955872, 76.3285904, -109.9773636, 84.6507568, -183.8462982, 186.3059082
5: -88.7699814, 69.1550369, -98.5042725, 76.8132401, -165.5832214, 167.6593018
6: -85.2733154, 82.2249146, -94.5402222, 91.1772690, -176.4505615, 176.7651062
7: -92.9629669, 78.0268402, -103.2613220, 86.5708923, -179.5338593, 181.2880859
8: -112.5315170, 77.0708542, -124.6104965, 85.2770386, -197.8085480, 201.6813354
9: -84.4136353, 83.4204712, -93.7517166, 92.5160675, -176.9297028, 177.1721802

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1969197, upper bound: 193.1899251
time: 7.46 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1817386, upper bound: 193.1894454
time: 5.27 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -102.6823044, 81.7061234, -92.5710983, 73.7949295, -176.4772186, 174.2771912
1: -86.5658798, 72.6101608, -77.9647217, 65.4957733, -152.0616455, 150.5748901
2: -113.4187164, 73.6618729, -102.2964478, 66.4200058, -179.8387146, 175.9582977
3: -120.3311996, 64.0238419, -108.5205307, 57.7372856, -178.0684814, 172.5443726
4: -109.9954300, 84.6645660, -99.1955872, 76.3285904, -186.3239899, 183.8601379
5: -98.5205078, 76.8254395, -88.7699814, 69.1550369, -167.6755371, 165.5954132
6: -94.5557632, 91.1921463, -85.2733154, 82.2249146, -176.7806396, 176.4654083
7: -103.2783661, 86.5848770, -92.9629669, 78.0268402, -181.3051453, 179.5478516
8: -124.6306076, 85.2908020, -112.5315170, 77.0708542, -201.7014465, 197.8223267
9: -93.7667542, 92.5312347, -84.4136353, 83.4204712, -177.1872253, 176.9448700

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1899251, upper bound: 193.2063316
time: 6.87 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1894454, upper bound: 193.2047223
time: 7.24 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -102.6823044, 81.7061234, -102.6823044, 81.7061234, -184.3884277, 184.3884277
1: -86.5658798, 72.6101608, -86.5658798, 72.6101608, -159.1760406, 159.1760406
2: -113.4187164, 73.6618729, -113.4187164, 73.6618729, -187.0805969, 187.0805969
3: -120.3311996, 64.0238419, -120.3311996, 64.0238419, -184.3550415, 184.3550415
4: -109.9954300, 84.6645660, -109.9954300, 84.6645660, -194.6599731, 194.6599731
5: -98.5205078, 76.8254395, -98.5205078, 76.8254395, -175.3459320, 175.3459320
6: -94.5557632, 91.1921463, -94.5557632, 91.1921463, -185.7478790, 185.7478790
7: -103.2783661, 86.5848770, -103.2783661, 86.5848770, -189.8632355, 189.8632355
8: -124.6306076, 85.2908020, -124.6306076, 85.2908020, -209.9214172, 209.9214172
9: -93.7667542, 92.5312347, -93.7667542, 92.5312347, -186.2979889, 186.2979889

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2063472, upper bound: 193.2533588
time: 8.23 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2609467, upper bound: 193.2857393
time: 7.20 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 17.21 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 17.21
Output dim: 2, lower bound: -193.1779398, upper bound: 193.1906249
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 17.21
Output dim: 2, lower bound: -193.2587072, upper bound: 193.2587072
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 17.21
Output dim: 2, lower bound: -193.1969197, upper bound: 193.1899251
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 17.21
Output dim: 2, lower bound: -193.1817386, upper bound: 193.1894454
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.21
Output dim: 2, lower bound: -193.1899251, upper bound: 193.2063316
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.21
Output dim: 2, lower bound: -193.1894454, upper bound: 193.2047223
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.21
Output dim: 2, lower bound: -193.2063472, upper bound: 193.2533588
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.21
Output dim: 2, lower bound: -193.2609467, upper bound: 193.2857393

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -89.2959442, 71.2327652, -86.8312302, 69.3837051, -158.6795959, 158.0639954
1: -75.2216873, 63.2028046, -73.1511383, 61.5169716, -136.7386475, 136.3539124
2: -98.7160873, 64.1070404, -96.1174850, 62.3816605, -161.0977020, 160.2244720
3: -104.6967621, 55.6746979, -101.9138184, 54.0041809, -158.7009430, 157.5884552
4: -95.6990967, 73.6709442, -93.1746292, 71.7173920, -167.4164581, 166.8455811
5: -85.6522598, 66.7137909, -83.3733978, 64.8451233, -150.4973755, 150.0871887
6: -82.2726593, 79.3452911, -80.0860214, 77.2245026, -159.4971619, 159.4313049
7: -89.6857605, 75.3030243, -87.2716370, 73.2969360, -162.9826965, 162.5746460
8: -108.6165161, 74.3738708, -105.7937851, 72.3115616, -180.9280701, 180.1676483
9: -81.4572220, 80.5020981, -79.2923431, 78.3324738, -159.7896881, 159.7944336

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 194

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1385198, upper bound: 193.1553652
time: 6.95 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1382614, upper bound: 193.1546395
time: 7.54 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -92.5710983, 73.7949295, -91.3974533, 72.8763275, -165.4474030, 165.1923828
1: -77.9647217, 65.4957733, -76.9810562, 64.6745911, -142.6393127, 142.4768372
2: -102.2964478, 66.4200058, -101.0148010, 65.5897446, -167.8861847, 167.4347687
3: -108.5205307, 57.7372856, -107.1548767, 56.9989738, -165.5195007, 164.8921356
4: -99.1955872, 76.3285904, -97.9424896, 75.3752289, -174.5707855, 174.2710876
5: -88.7699814, 69.1550369, -87.6518936, 68.2782364, -157.0481720, 156.8069305
6: -85.2733154, 82.2249146, -84.1981201, 81.1932297, -166.4665375, 166.4229736
7: -92.9629669, 78.0268402, -91.7902985, 77.0505829, -170.0135498, 169.8170624
8: -112.5315170, 77.0708542, -111.1318130, 76.1022568, -188.6337280, 188.2026672
9: -84.4136353, 83.4204712, -83.3529892, 82.3756104, -166.7892303, 166.7734680

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1862465, upper bound: 193.1965304
time: 7.41 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1814938, upper bound: 193.1814938
time: 5.57 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -91.0616760, 72.6025925, -93.1979141, 74.2191696, -165.2808533, 165.8004456
1: -76.6826172, 64.4346008, -78.5150757, 65.9419022, -142.6244965, 142.9496765
2: -100.6281509, 65.3385468, -102.9403305, 66.8650208, -167.4931641, 168.2788696
3: -106.7591019, 56.8005905, -109.2613983, 58.1358948, -164.8949585, 166.0619812
4: -97.5726166, 75.0812912, -99.8009338, 76.8330994, -174.4057159, 174.8822021
5: -87.3083038, 68.0207214, -89.3356705, 69.6916885, -156.9999695, 157.3563843
6: -83.8796844, 80.8764267, -85.8030701, 82.7228622, -166.6025391, 166.6795044
7: -91.4320068, 76.7481995, -93.6550293, 78.5517502, -169.9837494, 170.4032288
8: -110.7104263, 75.8192291, -113.1935501, 77.4376144, -188.1480408, 189.0127716
9: -83.0200348, 82.0559158, -85.0071716, 83.9579544, -166.9779968, 167.0630646

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2046459, upper bound: 193.1893511
time: 7.76 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2046459, upper bound: 193.1893511
time: 7.75 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -84.9166107, 67.7433167, -95.7003632, 76.2059479, -161.1225433, 163.4436798
1: -71.4922867, 60.1197433, -80.5826874, 67.6943817, -139.1866760, 140.7024231
2: -93.8546295, 60.9453583, -105.7627716, 68.6435928, -162.4982147, 166.7081299
3: -99.6206512, 52.9888458, -112.2887344, 59.6110573, -159.2317047, 165.2775726
4: -90.9589005, 70.0375290, -102.4713440, 78.8884888, -169.8473511, 172.5088806
5: -81.3705750, 63.4064674, -91.7461395, 71.4958878, -152.8664551, 155.1526031
6: -78.2182312, 75.4004517, -88.0956955, 84.8945084, -163.1127167, 163.4961548
7: -85.2451477, 71.5695801, -96.1814346, 80.6564178, -165.9015656, 167.7510071
8: -103.3178558, 70.7129364, -116.2971878, 79.4014053, -182.7192688, 187.0101318
9: -77.3701935, 76.5061646, -87.2306671, 86.1436691, -163.5138550, 163.7368317

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2047223, upper bound: 193.1894454
time: 6.82 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2047223, upper bound: 193.1894454
time: 7.80 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -93.2157745, 74.2332306, -91.0616760, 72.6025925, -165.8183289, 165.2949066
1: -78.5291367, 65.9542618, -76.6826172, 64.4346008, -142.9637451, 142.6368561
2: -102.9597778, 66.8783493, -100.6281509, 65.3385468, -168.2983246, 167.5065002
3: -109.2818756, 58.1466141, -106.7591019, 56.8005905, -166.0824585, 164.9056702
4: -99.8193741, 76.8472061, -97.5726166, 75.0812912, -174.9006653, 174.4197998
5: -89.3522644, 69.7041397, -87.3083038, 68.0207214, -157.3729858, 157.0124054
6: -85.8189316, 82.7380447, -83.8796844, 80.8764267, -166.6953583, 166.6177368
7: -93.6724472, 78.5660324, -91.4320068, 76.7481995, -170.4206543, 169.9980164
8: -113.2140656, 77.4516983, -110.7104263, 75.8192291, -189.0332947, 188.1621246
9: -85.0225220, 83.9734421, -83.0200348, 82.0559158, -167.0784302, 166.9934692

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1893511, upper bound: 193.2046459
time: 6.91 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1893511, upper bound: 193.2046459
time: 6.93 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -95.7926102, 76.2786789, -84.9166107, 67.7433167, -163.5359192, 161.1952667
1: -80.6553192, 67.7581940, -71.4922867, 60.1197433, -140.7750549, 139.2504883
2: -105.8632736, 68.7125015, -93.8546295, 60.9453583, -166.8086243, 162.5671387
3: -112.3945236, 59.6664543, -99.6206512, 52.9888458, -165.3833618, 159.2870941
4: -102.5667572, 78.9613342, -90.9589005, 70.0375290, -172.6042786, 169.9202118
5: -91.8319016, 71.5602570, -81.3705750, 63.4064674, -155.2383728, 152.9308319
6: -88.1777344, 84.9729919, -78.2182312, 75.4004517, -163.5781860, 163.1911621
7: -96.2714691, 80.7301788, -85.2451477, 71.5695801, -167.8410492, 165.9753265
8: -116.4032059, 79.4741821, -103.3178558, 70.7129364, -187.1161499, 182.7920380
9: -87.3099365, 86.2237701, -77.3701935, 76.5061646, -163.8161011, 163.5939331

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1894454, upper bound: 193.2047223
time: 6.78 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1894454, upper bound: 193.2047223
time: 7.62 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -97.6332626, 77.8344498, -99.4943390, 79.2125931, -176.8458557, 177.3287506
1: -82.3228912, 69.1114731, -83.8975983, 70.3797684, -152.7026672, 153.0090637
2: -107.9961472, 70.1227875, -109.9354782, 71.4132309, -179.4093781, 180.0582581
3: -114.5261002, 60.7184143, -116.6112213, 62.0161781, -176.5422821, 177.3296356
4: -104.7124252, 80.6111679, -106.5949707, 82.0781479, -186.7905579, 187.2061462
5: -93.7768860, 73.0213013, -95.4867783, 74.4502106, -168.2270813, 168.5080719
6: -89.9974365, 86.7931061, -91.6359024, 88.3902130, -178.3876495, 178.4289856
7: -98.2770386, 82.4231186, -100.0902710, 83.9355545, -182.2125854, 182.5133972
8: -118.7136917, 81.0908432, -120.8204651, 82.6663818, -201.3800659, 201.9113007
9: -89.2590256, 88.0515442, -90.8889008, 89.6906433, -178.9496307, 178.9404449

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2311075, upper bound: 193.2208395
time: 7.30 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2285832, upper bound: 193.2200978
time: 6.78 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -101.4703140, 80.7576370, -102.6823044, 81.7061234, -183.1764374, 183.4399261
1: -85.5506668, 71.7622910, -86.5658798, 72.6101608, -158.1608276, 158.3281403
2: -112.0956573, 72.8048935, -113.4187164, 73.6618729, -185.7575378, 186.2236023
3: -118.9217606, 63.2614098, -120.3311996, 64.0238419, -182.9456024, 183.5926056
4: -108.7016373, 83.6804581, -109.9954300, 84.6645660, -193.3662109, 193.6758728
5: -97.3661270, 75.9204025, -98.5205078, 76.8254395, -174.1915588, 174.4409027
6: -93.4459381, 90.1269226, -94.5557632, 91.1921463, -184.6380157, 184.6826782
7: -102.0670776, 85.5770569, -103.2783661, 86.5848770, -188.6519470, 188.8554230
8: -123.1853180, 84.2911224, -124.6306076, 85.2908020, -208.4761200, 208.9217072
9: -92.6712875, 91.4525681, -93.7667542, 92.5312347, -185.2025146, 185.2193146

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2134823, upper bound: 193.1979056
time: 8.35 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1900311, upper bound: 193.1900297
time: 5.58 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 15.28 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 2, lower bound: -193.1385198, upper bound: 193.1553652
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 2, lower bound: -193.1382614, upper bound: 193.1546395
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 2, lower bound: -193.1862465, upper bound: 193.1965304
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 2, lower bound: -193.1814938, upper bound: 193.1814938
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 2, lower bound: -193.2046459, upper bound: 193.1893511
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 2, lower bound: -193.2046459, upper bound: 193.1893511
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 2, lower bound: -193.2047223, upper bound: 193.1894454
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 2, lower bound: -193.2047223, upper bound: 193.1894454
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 2, lower bound: -193.1893511, upper bound: 193.2046459
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 2, lower bound: -193.1893511, upper bound: 193.2046459
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 2, lower bound: -193.1894454, upper bound: 193.2047223
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 2, lower bound: -193.1894454, upper bound: 193.2047223
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 2, lower bound: -193.2311075, upper bound: 193.2208395
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 2, lower bound: -193.2285832, upper bound: 193.2200978
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 2, lower bound: -193.2134823, upper bound: 193.1979056
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 2, lower bound: -193.1900311, upper bound: 193.1900297

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -86.5067673, 68.9808121, -86.0672379, 68.7665558, -155.2733154, 155.0480499
1: -72.8428116, 61.1990776, -72.4991608, 60.9672318, -133.8099823, 133.6982422
2: -95.6030197, 62.0976753, -95.2643967, 61.8312187, -157.4342346, 157.3620758
3: -101.3689728, 53.9085350, -101.0009384, 53.5208282, -154.8898010, 154.9094696
4: -92.6677475, 71.3833313, -92.3455963, 71.0907440, -163.7584839, 163.7289124
5: -82.9841766, 64.6000443, -82.6428604, 64.2658844, -147.2500610, 147.2428894
6: -79.6910095, 76.8342209, -79.3792419, 76.5364838, -156.2274933, 156.2134705
7: -86.8810120, 72.9648972, -86.5030060, 72.6556549, -159.5366669, 159.4678955
8: -105.1979065, 71.9992523, -104.8561707, 71.6620483, -176.8599548, 176.8554230
9: -78.9165726, 77.9636459, -78.5954742, 77.6370239, -156.5535736, 156.5591125

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1382614, upper bound: 193.1546395
time: 6.01 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1382614, upper bound: 193.1546395
time: 6.04 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -91.5171432, 72.8968201, -84.9198151, 67.8372879, -159.3544312, 157.8166199
1: -76.9946060, 64.6744614, -71.5165939, 60.1400223, -137.1345825, 136.1910400
2: -101.0883408, 65.6246109, -93.9826202, 61.0045433, -162.0928802, 159.6072388
3: -107.2600555, 56.8996353, -99.6311111, 52.7944565, -160.0545044, 156.5307465
4: -98.0548782, 75.5060959, -91.1018372, 70.1463547, -168.2012329, 166.6079407
5: -87.8042145, 68.2762451, -81.5439377, 63.3947220, -151.1989441, 149.8201752
6: -84.3204651, 81.2356186, -78.3185043, 75.5022812, -159.8227539, 159.5540924
7: -91.9283066, 77.1524277, -85.3513031, 71.6909256, -163.6192322, 162.5037231
8: -111.2315292, 75.9833603, -103.4426117, 70.6804886, -181.9120178, 179.4259644
9: -83.4994659, 82.4092865, -77.5491180, 76.5904999, -160.0899658, 159.9584045

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1382614, upper bound: 193.1546395
time: 5.89 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1382614, upper bound: 193.1546395
time: 5.95 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -84.0246582, 67.0425644, -89.8884506, 71.6841354, -155.7088013, 156.9309540
1: -70.7055969, 59.4865685, -75.6991653, 63.6136665, -134.3192596, 135.1857300
2: -92.8490372, 60.2935104, -99.3468628, 64.5084534, -157.3574829, 159.6403809
3: -98.5477219, 52.4346275, -105.3938141, 56.0625610, -154.6102753, 157.8284302
4: -90.0032654, 69.2684631, -96.3198471, 74.1283417, -164.1315918, 165.5883179
5: -80.4947510, 62.7323189, -86.1904678, 67.1440811, -147.6388245, 148.9227753
6: -77.3808975, 74.5888672, -82.8048172, 79.8450394, -157.2259369, 157.3936768
7: -84.2931061, 70.7854767, -90.2597427, 75.7720795, -160.0651855, 161.0452118
8: -102.2184448, 69.9830246, -109.3110199, 74.8509216, -177.0693665, 179.2940369
9: -76.5220490, 75.6940460, -81.9597244, 81.0112457, -157.5332947, 157.6537628

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1814938, upper bound: 193.1814938
time: 5.45 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1814938, upper bound: 193.1814938
time: 5.45 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -82.6786575, 65.9964752, -83.7462158, 66.8264008, -149.5050659, 149.7426758
1: -69.5295792, 58.5330467, -70.5111237, 59.2990532, -128.8286285, 129.0441437
2: -91.4361420, 59.3242874, -92.5756760, 60.1167831, -151.5529175, 151.8999481
3: -97.0741730, 51.5299339, -98.2572250, 52.2534904, -149.3276520, 149.7871399
4: -88.5501251, 68.1753769, -89.7111359, 69.0877075, -157.6378326, 157.8865051
5: -79.1940460, 61.6374016, -80.2563553, 62.5314407, -141.7254944, 141.8937531
6: -76.1422424, 73.3440933, -77.1461563, 74.3714981, -150.5137177, 150.4902496
7: -82.9286499, 69.6382446, -84.0750732, 70.5943680, -153.5230103, 153.7132874
8: -100.6930008, 68.8089676, -101.9204025, 69.7478561, -170.4408417, 170.7293701
9: -75.1964569, 74.4152985, -76.3110657, 75.4645615, -150.6609802, 150.7263641

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1814938, upper bound: 193.1814938
time: 4.85 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1814938, upper bound: 193.1814938
time: 5.23 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -84.0246582, 67.0425644, -93.1979141, 74.2191696, -158.2438354, 160.2404175
1: -70.7055969, 59.4865685, -78.5150757, 65.9419022, -136.6474762, 138.0016479
2: -92.8490372, 60.2935104, -102.9403305, 66.8650208, -159.7140503, 163.2338409
3: -98.5477219, 52.4346275, -109.2613983, 58.1358948, -156.6835785, 161.6960297
4: -90.0032654, 69.2684631, -99.8009338, 76.8330994, -166.8363647, 169.0693817
5: -80.4947510, 62.7323189, -89.3356705, 69.6916885, -150.1864319, 152.0679932
6: -77.3808975, 74.5888672, -85.8030701, 82.7228622, -160.1037598, 160.3919373
7: -84.2931061, 70.7854767, -93.6550293, 78.5517502, -162.8448486, 164.4405060
8: -102.2184448, 69.9830246, -113.1935501, 77.4376144, -179.6560669, 183.1765594
9: -76.5220490, 75.6940460, -85.0071716, 83.9579544, -160.4800110, 160.7012177

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1372254, upper bound: 193.1278665
time: 7.54 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2063316, upper bound: 193.1899251
time: 7.50 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -82.6786575, 65.9964752, -93.1979141, 74.2191696, -156.8978271, 159.1943359
1: -69.5295792, 58.5330467, -78.5150757, 65.9419022, -135.4714508, 137.0481262
2: -91.4361420, 59.3242874, -102.9403305, 66.8650208, -158.3011627, 162.2646179
3: -97.0741730, 51.5299339, -109.2613983, 58.1358948, -155.2100525, 160.7913208
4: -88.5501251, 68.1753769, -99.8009338, 76.8330994, -165.3832245, 167.9762878
5: -79.1940460, 61.6374016, -89.3356705, 69.6916885, -148.8857269, 150.9730682
6: -76.1422424, 73.3440933, -85.8030701, 82.7228622, -158.8650665, 159.1471558
7: -82.9286499, 69.6382446, -93.6550293, 78.5517502, -161.4804077, 163.2932739
8: -100.6930008, 68.8089676, -113.1935501, 77.4376144, -178.1306152, 182.0025177
9: -75.1964569, 74.4152985, -85.0071716, 83.9579544, -159.1544037, 159.4224701

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 131

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1600232, upper bound: 193.1386722
time: 7.24 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1593259, upper bound: 193.1384222
time: 7.47 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -84.0237274, 67.0419388, -95.7003632, 76.2059479, -160.2296753, 162.7422791
1: -70.7048492, 59.4861183, -80.5826874, 67.6943817, -138.3992310, 140.0687866
2: -92.8481064, 60.2929192, -105.7627716, 68.6435928, -161.4916840, 166.0556946
3: -98.5468521, 52.4339104, -112.2887344, 59.6110573, -158.1579132, 164.7226410
4: -90.0018845, 69.2675934, -102.4713440, 78.8884888, -168.8903656, 171.7389374
5: -80.4937210, 62.7316856, -91.7461395, 71.4958878, -151.9896088, 154.4778290
6: -77.3799820, 74.5880508, -88.0956955, 84.8945084, -162.2744904, 162.6837463
7: -84.2921982, 70.7849045, -96.1814346, 80.6564178, -164.9486084, 166.9663239
8: -102.2175369, 69.9821243, -116.2971878, 79.4014053, -181.6189423, 186.2792816
9: -76.5214157, 75.6931229, -87.2306671, 86.1436691, -162.6650848, 162.9237976

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1282575, upper bound: 193.1188118
time: 6.05 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2046459, upper bound: 193.1894454
time: 6.97 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -82.6786575, 65.9964752, -95.7003632, 76.2059479, -158.8845978, 161.6968231
1: -69.5295792, 58.5330467, -80.5826874, 67.6943817, -137.2239685, 139.1157074
2: -91.4361420, 59.3242874, -105.7627716, 68.6435928, -160.0797272, 165.0870514
3: -97.0741730, 51.5299339, -112.2887344, 59.6110573, -156.6852264, 163.8186493
4: -88.5501251, 68.1753769, -102.4713440, 78.8884888, -167.4385834, 170.6467285
5: -79.1940460, 61.6374016, -91.7461395, 71.4958878, -150.6899414, 153.3835449
6: -76.1422424, 73.3440933, -88.0956955, 84.8945084, -161.0367432, 161.4397888
7: -82.9286499, 69.6382446, -96.1814346, 80.6564178, -163.5850677, 165.8196564
8: -100.6930008, 68.8089676, -116.2971878, 79.4014053, -180.0944061, 185.1061554
9: -75.1964569, 74.4152985, -87.2306671, 86.1436691, -161.3401184, 161.6459656

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685271, upper bound: 193.1484012
time: 7.75 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1639534, upper bound: 193.1466901
time: 7.33 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -93.2157745, 74.2332306, -84.0246582, 67.0425644, -160.2583008, 158.2578888
1: -78.5291367, 65.9542618, -70.7055969, 59.4865685, -138.0156860, 136.6598358
2: -102.9597778, 66.8783493, -92.8490372, 60.2935104, -163.2532806, 159.7273865
3: -109.2818756, 58.1466141, -98.5477219, 52.4346275, -161.7165070, 156.6942902
4: -99.8193741, 76.8472061, -90.0032654, 69.2684631, -169.0878296, 166.8504639
5: -89.3522644, 69.7041397, -80.4947510, 62.7323189, -152.0845795, 150.1988678
6: -85.8189316, 82.7380447, -77.3808975, 74.5888672, -160.4078064, 160.1189423
7: -93.6724472, 78.5660324, -84.2931061, 70.7854767, -164.4579163, 162.8591156
8: -113.2140656, 77.4516983, -102.2184448, 69.9830246, -183.1970825, 179.6701355
9: -85.0225220, 83.9734421, -76.5220490, 75.6940460, -160.7165680, 160.4954834

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1278665, upper bound: 193.1372254
time: 6.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1899251, upper bound: 193.2063316
time: 6.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -93.2157745, 74.2332306, -82.6786575, 65.9964752, -159.2122192, 156.9118958
1: -78.5291367, 65.9542618, -69.5295792, 58.5330467, -137.0621643, 135.4838104
2: -102.9597778, 66.8783493, -91.4361420, 59.3242874, -162.2840424, 158.3144836
3: -109.2818756, 58.1466141, -97.0741730, 51.5299339, -160.8117981, 155.2207642
4: -99.8193741, 76.8472061, -88.5501251, 68.1753769, -167.9947510, 165.3973236
5: -89.3522644, 69.7041397, -79.1940460, 61.6374016, -150.9896698, 148.8981781
6: -85.8189316, 82.7380447, -76.1422424, 73.3440933, -159.1630249, 158.8802643
7: -93.6724472, 78.5660324, -82.9286499, 69.6382446, -163.3106995, 161.4946899
8: -113.2140656, 77.4516983, -100.6930008, 68.8089676, -182.0230408, 178.1446838
9: -85.0225220, 83.9734421, -75.1964569, 74.4152985, -159.4378204, 159.1698761

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 131

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1386722, upper bound: 193.1600232
time: 6.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1384222, upper bound: 193.1593259
time: 5.94 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -95.7926102, 76.2786789, -84.0237274, 67.0419388, -162.8345337, 160.3023987
1: -80.6553192, 67.7581940, -70.7048492, 59.4861183, -140.1414185, 138.4630432
2: -105.8632736, 68.7125015, -92.8481064, 60.2929192, -166.1561890, 161.5606079
3: -112.3945236, 59.6664543, -98.5468521, 52.4339104, -164.8284302, 158.2132874
4: -102.5667572, 78.9613342, -90.0018845, 69.2675934, -171.8343506, 168.9632263
5: -91.8319016, 71.5602570, -80.4937210, 62.7316856, -154.5635834, 152.0539703
6: -88.1777344, 84.9729919, -77.3799820, 74.5880508, -162.7657776, 162.3529510
7: -96.2714691, 80.7301788, -84.2921982, 70.7849045, -167.0563660, 165.0223694
8: -116.4032059, 79.4741821, -102.2175369, 69.9821243, -186.3853149, 181.6917114
9: -87.3099365, 86.2237701, -76.5214157, 75.6931229, -163.0030518, 162.7451630

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1188118, upper bound: 193.1282575
time: 7.06 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1894454, upper bound: 193.2047223
time: 7.13 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -95.7926102, 76.2786789, -82.6786575, 65.9964752, -161.7890778, 158.9573212
1: -80.6553192, 67.7581940, -69.5295792, 58.5330467, -139.1883392, 137.2877655
2: -105.8632736, 68.7125015, -91.4361420, 59.3242874, -165.1875305, 160.1486511
3: -112.3945236, 59.6664543, -97.0741730, 51.5299339, -163.9244537, 156.7406158
4: -102.5667572, 78.9613342, -88.5501251, 68.1753769, -170.7421265, 167.5114594
5: -91.8319016, 71.5602570, -79.1940460, 61.6374016, -153.4692993, 150.7543030
6: -88.1777344, 84.9729919, -76.1422424, 73.3440933, -161.5218201, 161.1151733
7: -96.2714691, 80.7301788, -82.9286499, 69.6382446, -165.9097137, 163.6588287
8: -116.4032059, 79.4741821, -100.6930008, 68.8089676, -185.2121735, 180.1671753
9: -87.3099365, 86.2237701, -75.1964569, 74.4152985, -161.7252350, 161.4201813

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1484733, upper bound: 193.1685789
time: 7.36 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1466901, upper bound: 193.1639534
time: 6.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -95.9801636, 76.5295334, -90.0129395, 71.7277374, -167.7079010, 166.5424500
1: -80.9190674, 67.9496231, -75.8477859, 63.7133217, -144.6323853, 143.7973938
2: -106.1705551, 68.9377441, -99.4601898, 64.6185913, -170.7891541, 168.3979034
3: -112.5965805, 59.6916809, -105.5447769, 56.1289520, -168.7254791, 165.2364502
4: -102.9363022, 79.2460785, -96.4022903, 74.2494049, -177.1856995, 175.6483612
5: -92.1755066, 71.7774734, -86.3040314, 67.3179932, -159.4934998, 158.0815125
6: -88.4717178, 85.3169632, -82.8852081, 79.9227753, -168.3945007, 168.2021790
7: -96.5999680, 81.0225830, -90.4697342, 75.9036484, -172.5036163, 171.4923096
8: -116.7205582, 79.7219238, -109.3859711, 74.8155594, -191.5361176, 189.1078644
9: -87.7318268, 86.5569305, -82.1316376, 81.1192932, -168.8511200, 168.6885529

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2285832, upper bound: 193.2200978
time: 8.26 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2285832, upper bound: 193.2200978
time: 7.54 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -90.6830521, 72.3348007, -92.3463364, 73.5797806, -164.2628326, 164.6811371
1: -76.4432373, 64.2322464, -77.7614212, 65.3416061, -141.7848511, 141.9936523
2: -100.3320465, 65.1499481, -102.0926132, 66.2718735, -166.6039124, 167.2425385
3: -106.4420547, 56.4059525, -108.3655167, 57.5034027, -163.9454193, 164.7714539
4: -97.2345047, 74.8959808, -98.8774643, 76.1588440, -173.3933105, 173.7734222
5: -87.0596313, 67.8082657, -88.5492554, 68.9950714, -156.0546875, 156.3575134
6: -83.5894089, 80.5978012, -85.0217056, 81.9383926, -165.5278015, 165.6194763
7: -91.2766266, 76.5644684, -92.8212967, 77.8610840, -169.1377106, 169.3857727
8: -110.3422852, 75.3100204, -112.2782593, 76.6315689, -186.9738312, 187.5882721
9: -82.8704987, 81.7746964, -84.1980743, 83.1547852, -166.0252838, 165.9727783

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 81

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2285832, upper bound: 193.2200978
time: 7.10 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2285832, upper bound: 193.2200978
time: 7.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -101.4703140, 80.7576370, -102.5355301, 81.5892105, -183.0595245, 183.2931519
1: -85.5506668, 71.7622910, -86.4417496, 72.5058289, -158.0564880, 158.2039948
2: -112.0956573, 72.8048935, -113.2559128, 73.5562210, -185.6518860, 186.0607758
3: -118.9217606, 63.2614098, -120.1571426, 63.9339752, -182.8557434, 183.4185333
4: -108.7016373, 83.6804581, -109.8358383, 84.5430908, -193.2447205, 193.5162964
5: -97.3661270, 75.9204025, -98.3789444, 76.7156296, -174.0817566, 174.2993469
6: -93.4459381, 90.1269226, -94.4195557, 91.0610580, -184.5069885, 184.5464783
7: -102.0670776, 85.5770569, -103.1289291, 86.4605789, -188.5276489, 188.7059937
8: -123.1853180, 84.2911224, -124.4521027, 85.1698837, -208.3551941, 208.7432251
9: -92.6712875, 91.4525681, -93.6315002, 92.3988724, -185.0701294, 185.0840454

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 194

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1873788, upper bound: 193.1701068
time: 6.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1845095, upper bound: 193.1691812
time: 7.23 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -101.0362473, 80.4124985, -120.8364029, 96.2331467, -197.2693939, 201.2489014
1: -85.1840897, 71.4542313, -101.6838226, 85.5485077, -170.7325897, 173.1380615
2: -111.6151581, 72.4933472, -133.4672699, 86.7114639, -198.3266144, 205.9605865
3: -118.4072037, 62.9962425, -141.9740753, 75.5098801, -193.9170837, 204.9703217
4: -108.2293396, 83.3216400, -129.6575012, 99.5849838, -207.8143311, 212.9791412
5: -96.9479294, 75.5968323, -116.0501862, 90.3964844, -187.3444214, 191.6469879
6: -93.0435181, 89.7392349, -111.4693298, 107.2153931, -200.2589111, 201.2085571
7: -101.6266861, 85.2105408, -121.4932251, 101.8809586, -203.5076447, 206.7037354
8: -122.6571732, 83.9335327, -146.5337372, 100.3153687, -222.9725037, 230.4672546
9: -92.2713318, 91.0613708, -110.1578598, 108.8898926, -201.1612244, 201.2192230

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1624095, upper bound: 193.1609799
time: 5.18 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1605729, upper bound: 193.1605741
time: 4.87 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 11.40 seconds
IS_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1382614, upper bound: 193.1546395
IS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1382614, upper bound: 193.1546395
IS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1382614, upper bound: 193.1546395
IS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1382614, upper bound: 193.1546395
IS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1814938, upper bound: 193.1814938
IS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1814938, upper bound: 193.1814938
IS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1814938, upper bound: 193.1814938
IS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1814938, upper bound: 193.1814938
IS_A1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1372254, upper bound: 193.1278665
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.2063316, upper bound: 193.1899251
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1600232, upper bound: 193.1386722
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1593259, upper bound: 193.1384222
IS_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1282575, upper bound: 193.1188118
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.2046459, upper bound: 193.1894454
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1685271, upper bound: 193.1484012
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1639534, upper bound: 193.1466901
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1278665, upper bound: 193.1372254
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1899251, upper bound: 193.2063316
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1386722, upper bound: 193.1600232
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1384222, upper bound: 193.1593259
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1188118, upper bound: 193.1282575
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1894454, upper bound: 193.2047223
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1484733, upper bound: 193.1685789
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1466901, upper bound: 193.1639534
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.2285832, upper bound: 193.2200978
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.2285832, upper bound: 193.2200978
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.2285832, upper bound: 193.2200978
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.2285832, upper bound: 193.2200978
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1873788, upper bound: 193.1701068
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1845095, upper bound: 193.1691812
IS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1624095, upper bound: 193.1609799
IS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 11.40
Output dim: 2, lower bound: -193.1605729, upper bound: 193.1605741

## BFS IS instance: IS_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -86.5067673, 68.9808121, -84.0830154, 67.1637115, -153.6704712, 153.0638123
1: -72.8428116, 61.1990776, -70.8056717, 59.5392685, -132.3820801, 132.0047302
2: -95.6030197, 62.0976753, -93.0486984, 60.4013214, -156.0043335, 155.1463776
3: -101.3689728, 53.9085350, -98.6298065, 52.2657547, -153.6347198, 152.5383453
4: -92.6677475, 71.3833313, -90.1924515, 69.4629440, -162.1306763, 161.5757751
5: -82.9841766, 64.6000443, -80.7456207, 62.7615013, -145.7456665, 145.3456726
6: -79.6910095, 76.8342209, -77.5435791, 74.7497787, -154.4407349, 154.3778076
7: -86.8810120, 72.9648972, -84.5065231, 70.9897385, -157.8707275, 157.4714203
8: -105.1979065, 71.9992523, -102.4207458, 69.9753723, -175.1732788, 174.4199982
9: -78.9165726, 77.9636459, -76.7849808, 75.8305511, -154.7471161, 154.7486267

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=194.859130859375
rel_dist={2: [-193.2888173876961, 193.28881738769616]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2062466, upper bound: 193.2009854
time: 8.19 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1982083, upper bound: 193.1982083
time: 6.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.59 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 14.59
Output dim: 2, lower bound: -193.2062466, upper bound: 193.2009854
IS_B2, status: Status.UNKNOWN, split count: 1, time: 14.59
Output dim: 2, lower bound: -193.1982083, upper bound: 193.1982083

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -106.9841232, 85.0895767, -106.8382950, 84.9734421, -191.9575653, 191.9278259
1: -90.1812515, 75.6211243, -90.0579300, 75.5174866, -165.6987305, 165.6790161
2: -118.1329117, 76.7262192, -117.9711609, 76.6212387, -194.7541351, 194.6973724
3: -125.3423386, 66.6745224, -125.1694031, 66.5852203, -191.9275360, 191.8439026
4: -114.6084290, 88.1751862, -114.4499283, 88.0545197, -202.6629486, 202.6251221
5: -102.6499710, 80.0399475, -102.5093689, 79.9308472, -182.5808105, 182.5493011
6: -98.4937134, 94.9893875, -98.3584061, 94.8591614, -193.3528748, 193.3477936
7: -107.6003113, 90.1861038, -107.4518433, 90.0626450, -197.6629333, 197.6379395
8: -129.7671661, 88.8023758, -129.5898438, 88.6822357, -218.4494019, 218.3922119
9: -97.6945114, 96.3775101, -97.5601730, 96.2459793, -193.9404907, 193.9376678

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1772955, upper bound: 193.1732742
time: 9.22 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1767169, upper bound: 193.1714967
time: 8.52 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -105.7677231, 84.1228027, -125.2631302, 99.7118149, -205.4795380, 209.3859253
1: -89.1547241, 74.7581253, -105.4000320, 88.6470718, -177.8017883, 180.1581421
2: -116.7863922, 75.8537750, -138.3164825, 89.8680344, -206.6544037, 214.1702576
3: -123.9017868, 65.9316330, -147.1298981, 78.2388382, -202.1406250, 213.0615082
4: -113.2857513, 87.1698914, -134.3992157, 103.1958237, -216.4815674, 221.5690918
5: -101.4781342, 79.1332779, -120.2966156, 93.7067947, -195.1849365, 199.4298859
6: -97.3658371, 93.9034271, -115.5219269, 111.1207504, -208.4865875, 209.4253540
7: -106.3674164, 89.1594238, -125.9393921, 105.5887756, -211.9561920, 215.0988159
8: -128.2868347, 87.8006363, -151.8200989, 103.9287186, -232.2155304, 239.6207275
9: -96.5750809, 95.2810974, -114.2007523, 112.8472824, -209.4223633, 209.4818420

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1694781, upper bound: 193.1688162
time: 6.74 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1686633, upper bound: 193.1686633
time: 5.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 15.94 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 15.94
Output dim: 2, lower bound: -193.1772955, upper bound: 193.1732742
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 15.94
Output dim: 2, lower bound: -193.1767169, upper bound: 193.1714967
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 15.94
Output dim: 2, lower bound: -193.1694781, upper bound: 193.1688162
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 15.94
Output dim: 2, lower bound: -193.1686633, upper bound: 193.1686633

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -97.7335205, 77.7851639, -100.5216141, 79.9858170, -177.7193298, 178.3067627
1: -82.3264923, 69.1167221, -84.6947479, 71.0755844, -153.4020386, 153.8114471
2: -107.9109879, 70.0978775, -110.9910278, 72.0952454, -180.0062256, 181.0888977
3: -114.5465164, 60.9306030, -117.7964935, 62.6627235, -177.2092285, 178.7270966
4: -104.6631012, 80.5348206, -107.6590881, 82.8372116, -187.5003052, 188.1938782
5: -93.6913605, 73.0837784, -96.3917465, 75.1804657, -168.8718262, 169.4755249
6: -89.9549561, 86.7273865, -92.5278244, 89.2176285, -179.1725616, 179.2552185
7: -98.2143173, 82.3519287, -101.0423813, 84.7127075, -182.9269867, 183.3943024
8: -118.6094818, 81.1402130, -121.9705963, 83.4496994, -202.0591736, 203.1107635
9: -89.1525650, 88.0159988, -91.7263870, 90.5360718, -179.6886292, 179.7423859

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1767169, upper bound: 193.1714967
time: 8.37 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1767169, upper bound: 193.1714967
time: 9.55 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -99.5822067, 79.2581482, -94.3897934, 75.1241455, -174.7063599, 173.6479492
1: -83.8394089, 70.4107513, -79.5352249, 66.7783661, -150.6177673, 149.9459839
2: -110.0202560, 71.4120712, -104.2426147, 67.7238998, -177.7441254, 175.6546631
3: -116.8099594, 62.0013313, -110.6983719, 58.8666306, -175.6765747, 172.6997070
4: -106.6313553, 82.0522690, -101.0548325, 77.8216553, -184.4530029, 183.1071014
5: -95.4676819, 74.3925934, -90.4829712, 70.5973434, -166.0650330, 164.8755646
6: -91.6464310, 88.3180389, -86.8873825, 83.7672501, -175.4136658, 175.2053986
7: -100.0812988, 83.9033585, -94.9175186, 79.5792618, -179.6605530, 178.8208618
8: -120.9290237, 82.5663528, -114.6016998, 78.3348389, -199.2638550, 197.1680603
9: -90.7711182, 89.6123886, -86.1273193, 85.0098572, -175.7809753, 175.7396698

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1767169, upper bound: 193.1714967
time: 8.55 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1767169, upper bound: 193.1714967
time: 8.44 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -99.4741364, 79.1528320, -116.5087433, 92.8002319, -192.2743530, 195.6615448
1: -83.8111801, 70.3321762, -97.9613876, 82.4964371, -166.3076172, 168.2935638
2: -109.8306961, 71.3437119, -128.6401520, 83.5948181, -193.4255066, 199.9838562
3: -116.5562744, 62.0225563, -136.9194031, 72.8108368, -189.3671112, 198.9419556
4: -106.5202103, 81.9715424, -124.9911499, 95.9659729, -202.4861450, 206.9626770
5: -95.3824615, 74.3989868, -111.8221588, 87.1218414, -182.5042877, 186.2211456
6: -91.5561371, 88.2824402, -107.4446945, 103.2992554, -194.8553925, 195.7270966
7: -99.9803772, 83.8282852, -117.0521774, 98.1739044, -198.1542816, 200.8804626
8: -120.6942291, 82.5865250, -141.2560883, 96.6799622, -217.3741913, 223.8426056
9: -90.7627945, 89.5917664, -106.1145020, 104.9365845, -195.6993561, 195.7062683

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1686633, upper bound: 193.1686633
time: 6.52 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1686633, upper bound: 193.1686633
time: 6.13 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -93.4469070, 74.3724365, -117.6251450, 93.6994629, -187.1463470, 191.9975586
1: -78.7400970, 66.1088562, -98.8698273, 83.2650986, -162.0051880, 164.9786530
2: -103.1959915, 67.0462570, -129.9440918, 84.3800659, -187.5760345, 196.9903564
3: -109.5824966, 58.2891998, -138.3335876, 73.4073563, -182.9898376, 196.6227875
4: -100.0309296, 77.0434036, -126.1797104, 96.8795090, -196.9104309, 203.2231140
5: -89.5736465, 69.8925705, -112.8877411, 87.8721924, -177.4458313, 182.7803040
6: -86.0125809, 82.9252472, -108.4474411, 104.2384644, -190.2510376, 191.3726654
7: -93.9602127, 78.7820282, -118.1803894, 99.0977936, -193.0579681, 196.9624176
8: -113.4510803, 77.5553665, -142.7002563, 97.4974594, -210.9485474, 220.2556152
9: -85.2606125, 84.1591568, -107.0558243, 105.8575897, -191.1181946, 191.2149811

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 160

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1686633, upper bound: 193.1686633
time: 6.59 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1686633, upper bound: 193.1686633
time: 5.14 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 13.07 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 13.07
Output dim: 2, lower bound: -193.1767169, upper bound: 193.1714967
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 13.07
Output dim: 2, lower bound: -193.1767169, upper bound: 193.1714967
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 13.07
Output dim: 2, lower bound: -193.1767169, upper bound: 193.1714967
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 13.07
Output dim: 2, lower bound: -193.1767169, upper bound: 193.1714967
IS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.07
Output dim: 2, lower bound: -193.1686633, upper bound: 193.1686633
IS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.07
Output dim: 2, lower bound: -193.1686633, upper bound: 193.1686633
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.07
Output dim: 2, lower bound: -193.1686633, upper bound: 193.1686633
IS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.07
Output dim: 2, lower bound: -193.1686633, upper bound: 193.1686633

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -97.7335205, 77.7851639, -97.5915756, 77.6721802, -175.4057007, 175.3767395
1: -82.3264923, 69.1167221, -82.2064819, 69.0158005, -151.3422852, 151.3231812
2: -107.9109879, 70.0978775, -107.7535629, 69.9956894, -177.9066620, 177.8514099
3: -114.5465164, 60.9306030, -114.3782120, 60.8437042, -175.3902130, 175.3088074
4: -104.6631012, 80.5348206, -104.5086975, 80.4173355, -185.0804291, 185.0435028
5: -93.6913605, 73.0837784, -93.5545502, 72.9775543, -166.6689148, 166.6383209
6: -89.9549561, 86.7273865, -89.8232651, 86.6006165, -176.5555573, 176.5506592
7: -98.2143173, 82.3519287, -98.0697556, 82.2317505, -180.4460297, 180.4216766
8: -118.6094818, 81.1402130, -118.4368515, 81.0233612, -199.6328430, 199.5770111
9: -89.1525650, 88.0159988, -89.0218124, 87.8879929, -177.0405579, 177.0378113

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1772955, upper bound: 193.1732742
time: 8.29 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1772955, upper bound: 193.1732742
time: 7.84 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -97.7335205, 77.7851639, -99.4425507, 79.1466675, -176.8801880, 177.2277222
1: -82.3264923, 69.1167221, -83.7213898, 70.3115082, -152.6379852, 152.8380585
2: -107.9109879, 70.0978775, -109.8650208, 71.3113708, -179.2223206, 179.9628754
3: -114.5465164, 60.9306030, -116.6446609, 61.9155121, -176.4620361, 177.5752563
4: -104.6631012, 80.5348206, -106.4798355, 81.9367676, -186.5998383, 187.0146484
5: -93.6913605, 73.0837784, -95.3328323, 74.2876968, -167.9790649, 168.4166107
6: -89.9549561, 86.7273865, -91.5167694, 88.1934586, -178.1484070, 178.2441559
7: -98.2143173, 82.3519287, -99.9388962, 83.7848663, -181.9991608, 182.2908173
8: -118.6094818, 81.1402130, -120.7589951, 82.4510345, -201.0605164, 201.8991852
9: -89.1525650, 88.0159988, -90.6427155, 89.4864273, -178.6389923, 178.6587219

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 81

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1772955, upper bound: 193.1732742
time: 8.22 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1772955, upper bound: 193.1732742
time: 8.76 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -99.5822067, 79.2581482, -97.5751648, 77.6590729, -177.2412720, 176.8333130
1: -83.8394089, 70.4107513, -82.1924973, 69.0039139, -152.8433075, 152.6032410
2: -110.0202560, 71.4120712, -107.7352676, 69.9838257, -180.0040588, 179.1472931
3: -116.8099594, 62.0013313, -114.3580017, 60.8333664, -177.6433105, 176.3593292
4: -106.6313553, 82.0522690, -104.4903107, 80.4036789, -187.0350342, 186.5425720
5: -95.4676819, 74.3925934, -93.5388412, 72.9652481, -168.4329224, 167.9314270
6: -91.6464310, 88.3180389, -89.8077316, 86.5857162, -178.2321472, 178.1257477
7: -100.0812988, 83.9033585, -98.0524826, 82.2178116, -182.2991028, 181.9558105
8: -120.9290237, 82.5663528, -118.4169464, 81.0100021, -201.9390259, 200.9832764
9: -90.7711182, 89.6123886, -89.0061646, 87.8733292, -178.6444397, 178.6185455

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 131

## Relational analysis of IS_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1410483, upper bound: 193.1352734
time: 8.69 seconds

## Relational analysis of IS_B1_A2_B1_B2

### Relational analysis result of IS_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1409077, upper bound: 193.1352280
time: 9.34 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -99.5822067, 79.2581482, -99.4425507, 79.1466675, -178.7288818, 178.7006989
1: -83.8394089, 70.4107513, -83.7213898, 70.3115082, -154.1509094, 154.1320953
2: -110.0202560, 71.4120712, -109.8650208, 71.3113708, -181.3315887, 181.2770386
3: -116.8099594, 62.0013313, -116.6446609, 61.9155121, -178.7254639, 178.6459961
4: -106.6313553, 82.0522690, -106.4798355, 81.9367676, -188.5681152, 188.5321045
5: -95.4676819, 74.3925934, -95.3328323, 74.2876968, -169.7553711, 169.7254333
6: -91.6464310, 88.3180389, -91.5167694, 88.1934586, -179.8398743, 179.8348083
7: -100.0812988, 83.9033585, -99.9388962, 83.7848663, -183.8661499, 183.8422089
8: -120.9290237, 82.5663528, -120.7589951, 82.4510345, -203.3800659, 203.3253479
9: -90.7711182, 89.6123886, -90.6427155, 89.4864273, -180.2575378, 180.2550964

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1767169, upper bound: 193.1714967
time: 9.88 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1767169, upper bound: 193.1714967
time: 9.98 seconds

## BFS IS instance: IS_B2_B1_A1

### Backsubstitution after applying IS history:
0: -96.5495529, 76.8430176, -116.5087433, 92.8002319, -189.3497772, 193.3517609
1: -81.3275528, 68.2761612, -97.9613876, 82.4964371, -163.8239899, 166.2375336
2: -106.5987091, 69.2477036, -128.6401520, 83.5948181, -190.1935272, 197.8878479
3: -113.1444931, 60.2065926, -136.9194031, 72.8108368, -185.9553223, 197.1259918
4: -103.3761368, 79.5561905, -124.9911499, 95.9659729, -199.3420868, 204.5473328
5: -92.5502243, 72.1997681, -111.8221588, 87.1218414, -179.6720581, 184.0219269
6: -88.8564987, 85.6703033, -107.4446945, 103.2992554, -192.1557465, 193.1149750
7: -97.0128403, 81.3515396, -117.0521774, 98.1739044, -195.1867371, 198.4037170
8: -117.1664963, 80.1641769, -141.2560883, 96.6799622, -213.8464661, 221.4202576
9: -88.0633850, 86.9484863, -106.1145020, 104.9365845, -192.9999695, 193.0629883

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 96

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_B2_B1_A1_B1

### Relational analysis result of IS_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1603383, upper bound: 193.1587629
time: 6.50 seconds

## Relational analysis of IS_B2_B1_A1_B2

### Relational analysis result of IS_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1595011, upper bound: 193.1585172
time: 8.12 seconds

## BFS IS instance: IS_B2_B1_A2

### Backsubstitution after applying IS history:
0: -98.4697723, 78.3705063, -116.5087433, 92.8002319, -191.2700043, 194.8792267
1: -82.9011688, 69.6211624, -97.9613876, 82.4964371, -165.3976135, 167.5825500
2: -108.7851257, 70.6120148, -128.6401520, 83.5948181, -192.3799286, 199.2521667
3: -115.4935303, 61.3190765, -136.9194031, 72.8108368, -188.3043671, 198.2384796
4: -105.4242020, 81.1339951, -124.9911499, 95.9659729, -201.3901367, 206.1251373
5: -94.3943558, 73.5598755, -111.8221588, 87.1218414, -181.5162048, 185.3820343
6: -90.6147614, 87.3250351, -107.4446945, 103.2992554, -193.9140167, 194.7696991
7: -98.9509048, 82.9618454, -117.0521774, 98.1739044, -197.1248169, 200.0140228
8: -119.5724716, 81.6467133, -141.2560883, 96.6799622, -216.2524414, 222.9027863
9: -89.7487106, 88.6090088, -106.1145020, 104.9365845, -194.6852875, 194.7235107

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_B2_B1_A2_B1

### Relational analysis result of IS_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1603383, upper bound: 193.1587629
time: 6.56 seconds

## Relational analysis of IS_B2_B1_A2_B2

### Relational analysis result of IS_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1595011, upper bound: 193.1585172
time: 6.80 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -96.5451584, 76.8395691, -117.6251450, 93.6994629, -190.2446289, 194.4647217
1: -81.3238754, 68.2729263, -98.8698273, 83.2650986, -164.5889740, 167.1427307
2: -106.5937653, 69.2447128, -129.9440918, 84.3800659, -190.9738159, 199.1888123
3: -113.1391373, 60.2037277, -138.3335876, 73.4073563, -186.5464935, 198.5373077
4: -103.3713608, 79.5524902, -126.1797104, 96.8795090, -200.2508698, 205.7322083
5: -92.5459366, 72.1965103, -112.8877411, 87.8721924, -180.4181213, 185.0842590
6: -88.8520126, 85.6662903, -108.4474411, 104.2384644, -193.0904846, 194.1137085
7: -97.0083237, 81.3478317, -118.1803894, 99.0977936, -196.1061096, 199.5281982
8: -117.1611252, 80.1606369, -142.7002563, 97.4974594, -214.6585846, 222.8609009
9: -88.0593109, 86.9444656, -107.0558243, 105.8575897, -193.9169006, 194.0002899

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 160

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B2_B2_A1_B1

### Relational analysis result of IS_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1686633, upper bound: 193.1686633
time: 6.04 seconds

## Relational analysis of IS_B2_B2_A1_B2

### Relational analysis result of IS_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1686633, upper bound: 193.1686633
time: 6.61 seconds

## BFS IS instance: IS_B2_B2_A2

### Backsubstitution after applying IS history:
0: -98.4697723, 78.3705063, -117.6251450, 93.6994629, -192.1692200, 195.9956360
1: -82.9011688, 69.6211624, -98.8698273, 83.2650986, -166.1662598, 168.4909668
2: -108.7851257, 70.6120148, -129.9440918, 84.3800659, -193.1651611, 200.5561066
3: -115.4935303, 61.3190765, -138.3335876, 73.4073563, -188.9008789, 199.6526489
4: -105.4242020, 81.1339951, -126.1797104, 96.8795090, -202.3036957, 207.3137054
5: -94.3943558, 73.5598755, -112.8877411, 87.8721924, -182.2665405, 186.4476166
6: -90.6147614, 87.3250351, -108.4474411, 104.2384644, -194.8532257, 195.7724609
7: -98.9509048, 82.9618454, -118.1803894, 99.0977936, -198.0486755, 201.1422119
8: -119.5724716, 81.6467133, -142.7002563, 97.4974594, -217.0699310, 224.3469391
9: -89.7487106, 88.6090088, -107.0558243, 105.8575897, -195.6062775, 195.6648254

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 96

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_B2_B2_A2_B1

### Relational analysis result of IS_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1597029, upper bound: 193.1585850
time: 8.37 seconds

## Relational analysis of IS_B2_B2_A2_B2

### Relational analysis result of IS_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1581978, upper bound: 193.1581978
time: 6.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 29.46 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 2, lower bound: -193.1772955, upper bound: 193.1732742
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 2, lower bound: -193.1772955, upper bound: 193.1732742
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 2, lower bound: -193.1772955, upper bound: 193.1732742
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 2, lower bound: -193.1772955, upper bound: 193.1732742
IS_B1_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 29.46
Output dim: 2, lower bound: -193.1410483, upper bound: 193.1352734
IS_B1_A2_B1_B2, status: Status.VERIFIED, split count: 4, time: 29.46
Output dim: 2, lower bound: -193.1409077, upper bound: 193.1352280
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 2, lower bound: -193.1767169, upper bound: 193.1714967
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 2, lower bound: -193.1767169, upper bound: 193.1714967
IS_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 2, lower bound: -193.1603383, upper bound: 193.1587629
IS_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 2, lower bound: -193.1595011, upper bound: 193.1585172
IS_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 2, lower bound: -193.1603383, upper bound: 193.1587629
IS_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 2, lower bound: -193.1595011, upper bound: 193.1585172
IS_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 2, lower bound: -193.1686633, upper bound: 193.1686633
IS_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 2, lower bound: -193.1686633, upper bound: 193.1686633
IS_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 2, lower bound: -193.1597029, upper bound: 193.1585850
IS_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 2, lower bound: -193.1581978, upper bound: 193.1581978

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -97.5915756, 77.6721802, -97.5915756, 77.6721802, -175.2637634, 175.2637634
1: -82.2064819, 69.0158005, -82.2064819, 69.0158005, -151.2222900, 151.2222900
2: -107.7535629, 69.9956894, -107.7535629, 69.9956894, -177.7492218, 177.7492218
3: -114.3782120, 60.8437042, -114.3782120, 60.8437042, -175.2219086, 175.2219086
4: -104.5086975, 80.4173355, -104.5086975, 80.4173355, -184.9260254, 184.9260254
5: -93.5545502, 72.9775543, -93.5545502, 72.9775543, -166.5320587, 166.5320587
6: -89.8232651, 86.6006165, -89.8232651, 86.6006165, -176.4238892, 176.4238892
7: -98.0697556, 82.2317505, -98.0697556, 82.2317505, -180.3014832, 180.3014832
8: -118.4368515, 81.0233612, -118.4368515, 81.0233612, -199.4602051, 199.4602051
9: -89.0218124, 87.8879929, -89.0218124, 87.8879929, -176.9098053, 176.9098053

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 208

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 208

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 217

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 217

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 131

## Relational analysis of IS_B1_A1_B1_A1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1484707, upper bound: 193.1453693
time: 9.25 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2

### Relational analysis result of IS_B1_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1430764, upper bound: 193.1368364
time: 8.27 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -116.5087433, 92.8002319, -97.5915756, 77.6721802, -194.1808929, 190.3918152
1: -97.9613876, 82.4964371, -82.2064819, 69.0158005, -166.9771881, 164.7029114
2: -128.6401520, 83.5948181, -107.7535629, 69.9956894, -198.6358337, 191.3483582
3: -136.9194031, 72.8108368, -114.3782120, 60.8437042, -197.7631073, 187.1890564
4: -124.9911499, 95.9659729, -104.5086975, 80.4173355, -205.4084778, 200.4746399
5: -111.8221588, 87.1218414, -93.5545502, 72.9775543, -184.7997131, 180.6763611
6: -107.4446945, 103.2992554, -89.8232651, 86.6006165, -194.0452728, 193.1225281
7: -117.0521774, 98.1739044, -98.0697556, 82.2317505, -199.2839203, 196.2436523
8: -141.2560883, 96.6799622, -118.4368515, 81.0233612, -222.2794495, 215.1168060
9: -106.1145020, 104.9365845, -89.0218124, 87.8879929, -194.0025024, 193.9584045

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 96

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 131

## Relational analysis of IS_B1_A1_B1_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1484707, upper bound: 193.1453693
time: 9.11 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2

### Relational analysis result of IS_B1_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1430764, upper bound: 193.1368364
time: 8.46 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -97.5915756, 77.6721802, -99.4425507, 79.1466675, -176.7382507, 177.1147308
1: -82.2064819, 69.0158005, -83.7213898, 70.3115082, -152.5179749, 152.7371674
2: -107.7535629, 69.9956894, -109.8650208, 71.3113708, -179.0648804, 179.8606873
3: -114.3782120, 60.8437042, -116.6446609, 61.9155121, -176.2937317, 177.4883728
4: -104.5086975, 80.4173355, -106.4798355, 81.9367676, -186.4454498, 186.8971710
5: -93.5545502, 72.9775543, -95.3328323, 74.2876968, -167.8422241, 168.3103638
6: -89.8232651, 86.6006165, -91.5167694, 88.1934586, -178.0167236, 178.1173706
7: -98.0697556, 82.2317505, -99.9388962, 83.7848663, -181.8545990, 182.1706085
8: -118.4368515, 81.0233612, -120.7589951, 82.4510345, -200.8878784, 201.7823486
9: -89.0218124, 87.8879929, -90.6427155, 89.4864273, -178.5082397, 178.5307007

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 81

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 131

## Relational analysis of IS_B1_A1_B2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1468707, upper bound: 193.1439122
time: 6.95 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2

### Relational analysis result of IS_B1_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1410282, upper bound: 193.1359494
time: 7.35 seconds

## BFS IS instance: IS_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -116.5087433, 92.8002319, -99.4425507, 79.1466675, -195.6554108, 192.2427826
1: -97.9613876, 82.4964371, -83.7213898, 70.3115082, -168.2728882, 166.2178345
2: -128.6401520, 83.5948181, -109.8650208, 71.3113708, -199.9514923, 193.4598236
3: -136.9194031, 72.8108368, -116.6446609, 61.9155121, -198.8349152, 189.4555054
4: -124.9911499, 95.9659729, -106.4798355, 81.9367676, -206.9278870, 202.4457855
5: -111.8221588, 87.1218414, -95.3328323, 74.2876968, -186.1098633, 182.4546661
6: -107.4446945, 103.2992554, -91.5167694, 88.1934586, -195.6381226, 194.8160248
7: -117.0521774, 98.1739044, -99.9388962, 83.7848663, -200.8370361, 198.1127777
8: -141.2560883, 96.6799622, -120.7589951, 82.4510345, -223.7071228, 217.4389648
9: -106.1145020, 104.9365845, -90.6427155, 89.4864273, -195.6009216, 195.5792999

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B1_A1_B2_A2_A1

### Relational analysis result of IS_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1673784, upper bound: 193.1649221
time: 8.25 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2

### Relational analysis result of IS_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1670132, upper bound: 193.1637969
time: 8.03 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -99.4425507, 79.1466675, -99.4425507, 79.1466675, -178.5892181, 178.5892181
1: -83.7213898, 70.3115082, -83.7213898, 70.3115082, -154.0328674, 154.0328674
2: -109.8650208, 71.3113708, -109.8650208, 71.3113708, -181.1763458, 181.1763458
3: -116.6446609, 61.9155121, -116.6446609, 61.9155121, -178.5601807, 178.5601807
4: -106.4798355, 81.9367676, -106.4798355, 81.9367676, -188.4165955, 188.4165955
5: -95.3328323, 74.2876968, -95.3328323, 74.2876968, -169.6205292, 169.6205292
6: -91.5167694, 88.1934586, -91.5167694, 88.1934586, -179.7102203, 179.7102203
7: -99.9388962, 83.7848663, -99.9388962, 83.7848663, -183.7237396, 183.7237396
8: -120.7589951, 82.4510345, -120.7589951, 82.4510345, -203.2100220, 203.2100220
9: -90.6427155, 89.4864273, -90.6427155, 89.4864273, -180.1291504, 180.1291504

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 208

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 208

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 131

## Relational analysis of IS_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1410483, upper bound: 193.1352734
time: 7.99 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1409077, upper bound: 193.1352282
time: 7.78 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -117.5896149, 93.6716385, -99.4425507, 79.1466675, -196.7362823, 193.1141968
1: -98.8401184, 83.2390518, -83.7213898, 70.3115082, -169.1516266, 166.9604340
2: -129.9041595, 84.3558960, -109.8650208, 71.3113708, -201.2154999, 194.2209015
3: -138.2902985, 73.3841934, -116.6446609, 61.9155121, -200.2058105, 190.0288544
4: -126.1411591, 96.8497696, -106.4798355, 81.9367676, -208.0779266, 203.3296051
5: -112.8532562, 87.8460464, -95.3328323, 74.2876968, -187.1409607, 183.1788483
6: -108.4114227, 104.2061157, -91.5167694, 88.1934586, -196.6048737, 195.7228851
7: -118.1439209, 99.0678711, -99.9388962, 83.7848663, -201.9287720, 199.0067291
8: -142.6570740, 97.4688568, -120.7589951, 82.4510345, -225.1081085, 218.2278442
9: -107.0229416, 105.8250580, -90.6427155, 89.4864273, -196.5093689, 196.4677734

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 96

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B1_A2_B2_A2_A1

### Relational analysis result of IS_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1666556, upper bound: 193.1636326
time: 7.22 seconds

## Relational analysis of IS_B1_A2_B2_A2_A2

### Relational analysis result of IS_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1658531, upper bound: 193.1610122
time: 9.33 seconds

## BFS IS instance: IS_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -94.5322952, 75.2450562, -113.6805038, 90.5612411, -185.0935059, 188.9255219
1: -79.6333160, 66.8534164, -95.5847778, 80.5018921, -160.1351776, 162.4381714
2: -104.3594894, 67.8111115, -125.5039368, 81.5810089, -185.9404907, 193.3150482
3: -110.7861252, 58.9601288, -133.6127167, 71.0623169, -181.8484039, 192.5728455
4: -101.1783981, 77.8896332, -121.9172363, 93.6309738, -194.8093719, 199.8068695
5: -90.6221466, 70.6905060, -109.1181183, 85.0058365, -175.6279907, 179.8086243
6: -86.9883499, 83.8727722, -104.8274841, 100.7821274, -187.7704773, 188.7002563
7: -94.9550934, 79.6566010, -114.1703796, 95.7978668, -190.7529602, 193.8269806
8: -114.7075958, 78.4943924, -137.8143158, 94.3430099, -209.0505676, 216.3087006
9: -86.2165298, 85.1222610, -103.5280380, 102.3802948, -188.5968323, 188.6502838

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B2_B1_A1_B1_A1

### Relational analysis result of IS_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1672908, upper bound: 193.1672908
time: 5.17 seconds

## Relational analysis of IS_B2_B1_A1_B1_A2

### Relational analysis result of IS_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1672908, upper bound: 193.1672908
time: 6.64 seconds

## BFS IS instance: IS_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -93.1090927, 74.1174774, -123.7605515, 98.5870819, -191.6961517, 197.8780212
1: -78.4310684, 65.8432541, -103.9030304, 87.5616531, -165.9927216, 169.7462769
2: -102.7642899, 66.7985687, -136.5111237, 88.7525101, -191.5167999, 203.3096619
3: -109.1250763, 58.0762520, -145.4371948, 77.2332687, -186.3583374, 203.5134277
4: -99.6179352, 76.7067947, -132.5587006, 101.8416595, -201.4595947, 209.2654877
5: -89.2682419, 69.6190262, -118.8030777, 92.4541321, -181.7223511, 188.4221039
6: -85.6642456, 82.5872879, -114.0350494, 109.5590057, -195.2232361, 196.6223297
7: -93.4830322, 78.4578400, -124.1021652, 104.2337952, -197.7168274, 202.5599670
8: -112.9581451, 77.3030548, -149.9032593, 102.4464874, -215.4046173, 227.2063141
9: -84.9045868, 83.8168259, -112.6165466, 111.2194595, -196.1240540, 196.4333496

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 246
type: A, layer: 1, pos: 246
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B2_B1_A1_B2_A1

### Relational analysis result of IS_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1672908, upper bound: 193.1672908
time: 5.34 seconds

## Relational analysis of IS_B2_B1_A1_B2_A2

### Relational analysis result of IS_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1672908, upper bound: 193.1672908
time: 6.52 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 15.43 seconds
IS_B1_A1_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 15.43
Output dim: 2, lower bound: -193.1484707, upper bound: 193.1453693
IS_B1_A1_B1_A1_A2, status: Status.VERIFIED, split count: 5, time: 15.43
Output dim: 2, lower bound: -193.1430764, upper bound: 193.1368364
IS_B1_A1_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 15.43
Output dim: 2, lower bound: -193.1484707, upper bound: 193.1453693
IS_B1_A1_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 15.43
Output dim: 2, lower bound: -193.1430764, upper bound: 193.1368364
IS_B1_A1_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 15.43
Output dim: 2, lower bound: -193.1468707, upper bound: 193.1439122
IS_B1_A1_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 15.43
Output dim: 2, lower bound: -193.1410282, upper bound: 193.1359494
IS_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.43
Output dim: 2, lower bound: -193.1673784, upper bound: 193.1649221
IS_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 15.43
Output dim: 2, lower bound: -193.1670132, upper bound: 193.1637969
IS_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 15.43
Output dim: 2, lower bound: -193.1410483, upper bound: 193.1352734
IS_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 15.43
Output dim: 2, lower bound: -193.1409077, upper bound: 193.1352282
IS_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.43
Output dim: 2, lower bound: -193.1666556, upper bound: 193.1636326
IS_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 15.43
Output dim: 2, lower bound: -193.1658531, upper bound: 193.1610122
IS_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.43
Output dim: 2, lower bound: -193.1672908, upper bound: 193.1672908
IS_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.43
Output dim: 2, lower bound: -193.1672908, upper bound: 193.1672908
IS_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.43
Output dim: 2, lower bound: -193.1672908, upper bound: 193.1672908
IS_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.43
Output dim: 2, lower bound: -193.1672908, upper bound: 193.1672908
IS_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.43
Output dim: 2, lower bound: -193.1603383, upper bound: 193.1587629
IS_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.43
Output dim: 2, lower bound: -193.1595011, upper bound: 193.1585172
IS_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.43
Output dim: 2, lower bound: -193.1686633, upper bound: 193.1686633
IS_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.43
Output dim: 2, lower bound: -193.1686633, upper bound: 193.1686633
IS_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.43
Output dim: 2, lower bound: -193.1597029, upper bound: 193.1585850
IS_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.43
Output dim: 2, lower bound: -193.1581978, upper bound: 193.1581978
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=194.859130859375
rel_dist={2: [-193.2885465354435, 193.28854653544352]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1804.32 seconds
