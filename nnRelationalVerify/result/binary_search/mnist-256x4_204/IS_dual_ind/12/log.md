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
execution time: IAR + LP analysis = 1.29 + 9.05 = 10.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -193.2890147, upper bound: 193.2890147


# Binary Search by BASE starts (time budget: 2689.66 seconds, max iter: 100)

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
Binary search time: 36.16 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2653.50 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2733919, upper bound: 193.2629149
time: 6.54 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2858051, upper bound: 193.2858051
time: 6.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.42 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.42
Output dim: 2, lower bound: -193.2733919, upper bound: 193.2629149
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.42
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

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2587831, upper bound: 193.2587831
time: 6.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2587831, upper bound: 193.2629149
time: 5.87 seconds

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

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2629149, upper bound: 193.2733919
time: 6.97 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2629149, upper bound: 193.2858051
time: 6.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 14.84 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.84
Output dim: 2, lower bound: -193.2587831, upper bound: 193.2587831
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.84
Output dim: 2, lower bound: -193.2587831, upper bound: 193.2629149
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.84
Output dim: 2, lower bound: -193.2629149, upper bound: 193.2733919
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.84
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
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1907923, upper bound: 193.2092209
time: 6.00 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
time: 5.79 seconds

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

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1907923, upper bound: 193.2408343
time: 6.47 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1957217
time: 5.48 seconds

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
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1968562, upper bound: 193.2255166
time: 5.15 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1818086, upper bound: 193.2220970
time: 8.42 seconds

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

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1968562, upper bound: 193.2737717
time: 7.23 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1957217, upper bound: 193.2701781
time: 7.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 16.11 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.11
Output dim: 2, lower bound: -193.1907923, upper bound: 193.2092209
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.11
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.11
Output dim: 2, lower bound: -193.1907923, upper bound: 193.2408343
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.11
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1957217
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.11
Output dim: 2, lower bound: -193.1968562, upper bound: 193.2255166
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.11
Output dim: 2, lower bound: -193.1818086, upper bound: 193.2220970
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.11
Output dim: 2, lower bound: -193.1968562, upper bound: 193.2737717
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.11
Output dim: 2, lower bound: -193.1957217, upper bound: 193.2701781

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

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
time: 5.14 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
time: 5.02 seconds

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

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
time: 4.76 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
time: 5.70 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -84.0246582, 67.0425644, -102.6823044, 81.7061234, -165.7307739, 169.7248383
1: -70.7055969, 59.4865685, -86.5658798, 72.6101608, -143.3157654, 146.0524445
2: -92.8490372, 60.2935104, -113.4187164, 73.6618729, -166.5108948, 173.7122192
3: -98.5477219, 52.4346275, -120.3311996, 64.0238419, -162.5715332, 172.7658234
4: -90.0032654, 69.2684631, -109.9954300, 84.6645660, -174.6678314, 179.2638855
5: -80.4947510, 62.7323189, -98.5205078, 76.8254395, -157.3201904, 161.2528076
6: -77.3808975, 74.5888672, -94.5557632, 91.1921463, -168.5730286, 169.1446228
7: -84.2931061, 70.7854767, -103.2783661, 86.5848770, -170.8779755, 174.0638275
8: -102.2184448, 69.9830246, -124.6306076, 85.2908020, -187.5092468, 194.6135864
9: -76.5220490, 75.6940460, -93.7667542, 92.5312347, -169.0532837, 169.4608002

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2220557, upper bound: 193.1956838
time: 6.69 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2220557, upper bound: 193.1957217
time: 6.08 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -82.6786575, 65.9964752, -99.7875290, 79.4152222, -162.0938721, 165.7839813
1: -69.5295792, 58.5330467, -84.1171341, 70.5770187, -140.1065826, 142.6501465
2: -91.4361420, 59.3242874, -110.2250214, 71.5913162, -163.0274658, 169.5492706
3: -97.0741730, 51.5299339, -116.9647598, 62.2275848, -159.3017578, 168.4946899
4: -88.5501251, 68.1753769, -106.8792038, 82.2835236, -170.8336487, 175.0545807
5: -79.1940460, 61.6374016, -95.7231903, 74.6541290, -153.8481750, 157.3605804
6: -76.1422424, 73.3440933, -91.8864288, 88.6109314, -164.7531738, 165.2305298
7: -82.9286499, 69.6382446, -100.3616486, 84.1448364, -167.0734863, 169.9998779
8: -100.6930008, 68.8089676, -121.1431503, 82.8822403, -183.5752411, 189.9521179
9: -75.1964569, 74.4152985, -91.1052475, 89.9167328, -165.1131744, 165.5205383

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2220557, upper bound: 193.1956838
time: 6.58 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2220557, upper bound: 193.1957217
time: 6.32 seconds

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
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1956838, upper bound: 193.2220557
time: 7.09 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1956838, upper bound: 193.2220557
time: 5.47 seconds

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

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1957217, upper bound: 193.2220970
time: 5.49 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1957217, upper bound: 193.2220970
time: 5.88 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -93.2157745, 74.2332306, -102.6823044, 81.7061234, -174.9218903, 176.9155273
1: -78.5291367, 65.9542618, -86.5658798, 72.6101608, -151.1392822, 152.5201263
2: -102.9597778, 66.8783493, -113.4187164, 73.6618729, -176.6216431, 180.2970581
3: -109.2818756, 58.1466141, -120.3311996, 64.0238419, -173.3057098, 178.4777985
4: -99.8193741, 76.8472061, -109.9954300, 84.6645660, -184.4839478, 186.8426056
5: -89.3522644, 69.7041397, -98.5205078, 76.8254395, -166.1777039, 168.2246094
6: -85.8189316, 82.7380447, -94.5557632, 91.1921463, -177.0110779, 177.2938080
7: -93.6724472, 78.5660324, -103.2783661, 86.5848770, -180.2573242, 181.8443756
8: -113.2140656, 77.4516983, -124.6306076, 85.2908020, -198.5048676, 202.0822754
9: -85.0225220, 83.9734421, -93.7667542, 92.5312347, -177.5537567, 177.7401886

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2701744, upper bound: 193.2701781
time: 5.23 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2701744, upper bound: 193.2701781
time: 4.73 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -95.7926102, 76.2786789, -99.7875290, 79.4152222, -175.2078247, 176.0662079
1: -80.6553192, 67.7581940, -84.1171341, 70.5770187, -151.2323151, 151.8753204
2: -105.8632736, 68.7125015, -110.2250214, 71.5913162, -177.4545746, 178.9375305
3: -112.3945236, 59.6664543, -116.9647598, 62.2275848, -174.6221008, 176.6312103
4: -102.5667572, 78.9613342, -106.8792038, 82.2835236, -184.8502808, 185.8405457
5: -91.8319016, 71.5602570, -95.7231903, 74.6541290, -166.4860229, 167.2834167
6: -88.1777344, 84.9729919, -91.8864288, 88.6109314, -176.7886658, 176.8594208
7: -96.2714691, 80.7301788, -100.3616486, 84.1448364, -180.4163055, 181.0918274
8: -116.4032059, 79.4741821, -121.1431503, 82.8822403, -199.2854462, 200.6173096
9: -87.3099365, 86.2237701, -91.1052475, 89.9167328, -177.2266693, 177.3289795

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2701744, upper bound: 193.2701781
time: 5.75 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2701744, upper bound: 193.2701781
time: 5.53 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 12.59 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -193.1818086, upper bound: 193.1818086
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -193.2220557, upper bound: 193.1956838
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -193.2220557, upper bound: 193.1957217
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -193.2220557, upper bound: 193.1956838
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -193.2220557, upper bound: 193.1957217
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -193.1956838, upper bound: 193.2220557
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -193.1956838, upper bound: 193.2220557
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -193.1957217, upper bound: 193.2220970
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -193.1957217, upper bound: 193.2220970
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -193.2701744, upper bound: 193.2701781
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -193.2701744, upper bound: 193.2701781
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -193.2701744, upper bound: 193.2701781
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -193.2701744, upper bound: 193.2701781

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

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1530521, upper bound: 193.1824621
time: 6.15 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1494581, upper bound: 193.1726807
time: 6.93 seconds

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
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1530521, upper bound: 193.1824621
time: 5.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1494581, upper bound: 193.1726807
time: 6.19 seconds

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

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1433039, upper bound: 193.1570720
time: 5.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1369549, upper bound: 193.1369549
time: 4.71 seconds

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

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1433039, upper bound: 193.1570720
time: 5.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1369549, upper bound: 193.1369549
time: 5.26 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -84.0246582, 67.0425644, -93.2157745, 74.2332306, -158.2578888, 160.2583008
1: -70.7055969, 59.4865685, -78.5291367, 65.9542618, -136.6598358, 138.0156860
2: -92.8490372, 60.2935104, -102.9597778, 66.8783493, -159.7273865, 163.2532806
3: -98.5477219, 52.4346275, -109.2818756, 58.1466141, -156.6942902, 161.7165070
4: -90.0032654, 69.2684631, -99.8193741, 76.8472061, -166.8504639, 169.0878296
5: -80.4947510, 62.7323189, -89.3522644, 69.7041397, -150.1988678, 152.0845795
6: -77.3808975, 74.5888672, -85.8189316, 82.7380447, -160.1189423, 160.4078064
7: -84.2931061, 70.7854767, -93.6724472, 78.5660324, -162.8591156, 164.4579163
8: -102.2184448, 69.9830246, -113.2140656, 77.4516983, -179.6701355, 183.1970825
9: -76.5220490, 75.6940460, -85.0225220, 83.9734421, -160.4954834, 160.7165680

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2383384, upper bound: 193.2312422
time: 7.44 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2329036, upper bound: 193.2191599
time: 7.41 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -84.0246582, 67.0425644, -95.7926102, 76.2786789, -160.3033447, 162.8351440
1: -70.7055969, 59.4865685, -80.6553192, 67.7581940, -138.4637909, 140.1418762
2: -92.8490372, 60.2935104, -105.8632736, 68.7125015, -161.5615387, 166.1567688
3: -98.5477219, 52.4346275, -112.3945236, 59.6664543, -158.2141418, 164.8291473
4: -90.0032654, 69.2684631, -102.5667572, 78.9613342, -168.9645996, 171.8352203
5: -80.4947510, 62.7323189, -91.8319016, 71.5602570, -152.0550079, 154.5642242
6: -77.3808975, 74.5888672, -88.1777344, 84.9729919, -162.3538666, 162.7666016
7: -84.2931061, 70.7854767, -96.2714691, 80.7301788, -165.0232849, 167.0569458
8: -102.2184448, 69.9830246, -116.4032059, 79.4741821, -181.6926270, 186.3862152
9: -76.5220490, 75.6940460, -87.3099365, 86.2237701, -162.7458038, 163.0039825

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2383384, upper bound: 193.2312422
time: 6.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2329036, upper bound: 193.2191599
time: 7.42 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -82.6786575, 65.9964752, -93.2157745, 74.2332306, -156.9118958, 159.2122192
1: -69.5295792, 58.5330467, -78.5291367, 65.9542618, -135.4838104, 137.0621643
2: -91.4361420, 59.3242874, -102.9597778, 66.8783493, -158.3144836, 162.2840424
3: -97.0741730, 51.5299339, -109.2818756, 58.1466141, -155.2207642, 160.8117981
4: -88.5501251, 68.1753769, -99.8193741, 76.8472061, -165.3973236, 167.9947510
5: -79.1940460, 61.6374016, -89.3522644, 69.7041397, -148.8981781, 150.9896698
6: -76.1422424, 73.3440933, -85.8189316, 82.7380447, -158.8802643, 159.1630249
7: -82.9286499, 69.6382446, -93.6724472, 78.5660324, -161.4946899, 163.3106995
8: -100.6930008, 68.8089676, -113.2140656, 77.4516983, -178.1446838, 182.0230408
9: -75.1964569, 74.4152985, -85.0225220, 83.9734421, -159.1698761, 159.4378204

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2033785, upper bound: 193.1845182
time: 6.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1888702, upper bound: 193.1561272
time: 7.06 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -82.6786575, 65.9964752, -95.7926102, 76.2786789, -158.9573212, 161.7890778
1: -69.5295792, 58.5330467, -80.6553192, 67.7581940, -137.2877655, 139.1883392
2: -91.4361420, 59.3242874, -105.8632736, 68.7125015, -160.1486511, 165.1875305
3: -97.0741730, 51.5299339, -112.3945236, 59.6664543, -156.7406158, 163.9244537
4: -88.5501251, 68.1753769, -102.5667572, 78.9613342, -167.5114594, 170.7421265
5: -79.1940460, 61.6374016, -91.8319016, 71.5602570, -150.7543030, 153.4692993
6: -76.1422424, 73.3440933, -88.1777344, 84.9729919, -161.1151733, 161.5218201
7: -82.9286499, 69.6382446, -96.2714691, 80.7301788, -163.6588287, 165.9097137
8: -100.6930008, 68.8089676, -116.4032059, 79.4741821, -180.1671753, 185.2121735
9: -75.1964569, 74.4152985, -87.3099365, 86.2237701, -161.4201813, 161.7252350

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2033785, upper bound: 193.1845182
time: 6.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1888702, upper bound: 193.1561272
time: 7.46 seconds

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

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1591126, upper bound: 193.1982809
time: 5.99 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1567605, upper bound: 193.1917980
time: 7.73 seconds

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

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1591126, upper bound: 193.1982809
time: 6.11 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1567605, upper bound: 193.1917980
time: 6.54 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -95.7926102, 76.2786789, -84.0246582, 67.0425644, -162.8351440, 160.3033447
1: -80.6553192, 67.7581940, -70.7055969, 59.4865685, -140.1418762, 138.4637909
2: -105.8632736, 68.7125015, -92.8490372, 60.2935104, -166.1567841, 161.5615387
3: -112.3945236, 59.6664543, -98.5477219, 52.4346275, -164.8291473, 158.2141418
4: -102.5667572, 78.9613342, -90.0032654, 69.2684631, -171.8352203, 168.9645996
5: -91.8319016, 71.5602570, -80.4947510, 62.7323189, -154.5642242, 152.0550079
6: -88.1777344, 84.9729919, -77.3808975, 74.5888672, -162.7666016, 162.3538666
7: -96.2714691, 80.7301788, -84.2931061, 70.7854767, -167.0569458, 165.0232849
8: -116.4032059, 79.4741821, -102.2184448, 69.9830246, -186.3862152, 181.6926270
9: -87.3099365, 86.2237701, -76.5220490, 75.6940460, -163.0039825, 162.7458038

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1579065, upper bound: 193.1939902
time: 6.21 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1562345, upper bound: 193.1889877
time: 7.35 seconds

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
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1579065, upper bound: 193.1939902
time: 5.48 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1562345, upper bound: 193.1889877
time: 6.30 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -93.2157745, 74.2332306, -93.2157745, 74.2332306, -167.4490051, 167.4490051
1: -78.5291367, 65.9542618, -78.5291367, 65.9542618, -144.4833679, 144.4833679
2: -102.9597778, 66.8783493, -102.9597778, 66.8783493, -169.8381195, 169.8381195
3: -109.2818756, 58.1466141, -109.2818756, 58.1466141, -167.4284668, 167.4284668
4: -99.8193741, 76.8472061, -99.8193741, 76.8472061, -176.6665802, 176.6665802
5: -89.3522644, 69.7041397, -89.3522644, 69.7041397, -159.0563965, 159.0563965
6: -85.8189316, 82.7380447, -85.8189316, 82.7380447, -168.5569763, 168.5569763
7: -93.6724472, 78.5660324, -93.6724472, 78.5660324, -172.2384796, 172.2384796
8: -113.2140656, 77.4516983, -113.2140656, 77.4516983, -190.6657715, 190.6657715
9: -85.0225220, 83.9734421, -85.0225220, 83.9734421, -168.9959717, 168.9959717

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2626132, upper bound: 193.2681971
time: 6.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2608773, upper bound: 193.2631528
time: 6.28 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -93.2157745, 74.2332306, -95.7926102, 76.2786789, -169.4944305, 170.0258484
1: -78.5291367, 65.9542618, -80.6553192, 67.7581940, -146.2873230, 146.6095428
2: -102.9597778, 66.8783493, -105.8632736, 68.7125015, -171.6722717, 172.7416229
3: -109.2818756, 58.1466141, -112.3945236, 59.6664543, -168.9483185, 170.5411072
4: -99.8193741, 76.8472061, -102.5667572, 78.9613342, -178.7807007, 179.4139557
5: -89.3522644, 69.7041397, -91.8319016, 71.5602570, -160.9125214, 161.5360260
6: -85.8189316, 82.7380447, -88.1777344, 84.9729919, -170.7919159, 170.9157715
7: -93.6724472, 78.5660324, -96.2714691, 80.7301788, -174.4026184, 174.8374939
8: -113.2140656, 77.4516983, -116.4032059, 79.4741821, -192.6882477, 193.8549042
9: -85.0225220, 83.9734421, -87.3099365, 86.2237701, -171.2462921, 171.2833862

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2626132, upper bound: 193.2681971
time: 6.32 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2608773, upper bound: 193.2631528
time: 6.18 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -95.7926102, 76.2786789, -93.2157745, 74.2332306, -170.0258484, 169.4944305
1: -80.6553192, 67.7581940, -78.5291367, 65.9542618, -146.6095428, 146.2873230
2: -105.8632736, 68.7125015, -102.9597778, 66.8783493, -172.7416229, 171.6722717
3: -112.3945236, 59.6664543, -109.2818756, 58.1466141, -170.5411072, 168.9483185
4: -102.5667572, 78.9613342, -99.8193741, 76.8472061, -179.4139557, 178.7807007
5: -91.8319016, 71.5602570, -89.3522644, 69.7041397, -161.5360260, 160.9125214
6: -88.1777344, 84.9729919, -85.8189316, 82.7380447, -170.9157715, 170.7919159
7: -96.2714691, 80.7301788, -93.6724472, 78.5660324, -174.8374939, 174.4026184
8: -116.4032059, 79.4741821, -113.2140656, 77.4516983, -193.8549042, 192.6882477
9: -87.3099365, 86.2237701, -85.0225220, 83.9734421, -171.2833862, 171.2462921

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2615750, upper bound: 193.2650646
time: 5.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2601496, upper bound: 193.2601483
time: 5.15 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -95.7926102, 76.2786789, -95.7926102, 76.2786789, -172.0712738, 172.0712738
1: -80.6553192, 67.7581940, -80.6553192, 67.7581940, -148.4135132, 148.4135132
2: -105.8632736, 68.7125015, -105.8632736, 68.7125015, -174.5757751, 174.5757751
3: -112.3945236, 59.6664543, -112.3945236, 59.6664543, -172.0609741, 172.0609741
4: -102.5667572, 78.9613342, -102.5667572, 78.9613342, -181.5280914, 181.5280914
5: -91.8319016, 71.5602570, -91.8319016, 71.5602570, -163.3921509, 163.3921509
6: -88.1777344, 84.9729919, -88.1777344, 84.9729919, -173.1507111, 173.1507111
7: -96.2714691, 80.7301788, -96.2714691, 80.7301788, -177.0016479, 177.0016479
8: -116.4032059, 79.4741821, -116.4032059, 79.4741821, -195.8773804, 195.8773804
9: -87.3099365, 86.2237701, -87.3099365, 86.2237701, -173.5336914, 173.5336914

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2615750, upper bound: 193.2650646
time: 6.09 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2601496, upper bound: 193.2601483
time: 5.32 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 12.74 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.1530521, upper bound: 193.1824621
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.1494581, upper bound: 193.1726807
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.1530521, upper bound: 193.1824621
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.1494581, upper bound: 193.1726807
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.1433039, upper bound: 193.1570720
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.1369549, upper bound: 193.1369549
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.1433039, upper bound: 193.1570720
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.1369549, upper bound: 193.1369549
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.2383384, upper bound: 193.2312422
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.2329036, upper bound: 193.2191599
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.2383384, upper bound: 193.2312422
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.2329036, upper bound: 193.2191599
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.2033785, upper bound: 193.1845182
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.1888702, upper bound: 193.1561272
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.2033785, upper bound: 193.1845182
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.1888702, upper bound: 193.1561272
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.1591126, upper bound: 193.1982809
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.1567605, upper bound: 193.1917980
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.1591126, upper bound: 193.1982809
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.1567605, upper bound: 193.1917980
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.1579065, upper bound: 193.1939902
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.1562345, upper bound: 193.1889877
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.1579065, upper bound: 193.1939902
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.1562345, upper bound: 193.1889877
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.2626132, upper bound: 193.2681971
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.2608773, upper bound: 193.2631528
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.2626132, upper bound: 193.2681971
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.2608773, upper bound: 193.2631528
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.2615750, upper bound: 193.2650646
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.2601496, upper bound: 193.2601483
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.2615750, upper bound: 193.2650646
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.74
Output dim: 2, lower bound: -193.2601496, upper bound: 193.2601483

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -81.2378311, 64.8312988, -84.0246582, 67.0425644, -148.2803345, 148.8559418
1: -68.3837051, 57.5365982, -70.7055969, 59.4865685, -127.8702698, 128.2421875
2: -89.7731781, 58.3186913, -92.8490372, 60.2935104, -150.0666809, 151.1677094
3: -95.2794189, 50.7288666, -98.5477219, 52.4346275, -147.7140045, 149.2765503
4: -87.0173950, 66.9982758, -90.0032654, 69.2684631, -156.2858582, 157.0015411
5: -77.8213120, 60.6631432, -80.4947510, 62.7323189, -140.5536041, 141.1578979
6: -74.8220978, 72.1273575, -77.3808975, 74.5888672, -149.4109650, 149.5082550
7: -81.5043869, 68.4531937, -84.2931061, 70.7854767, -152.2898254, 152.7462921
8: -98.8666458, 67.7094650, -102.2184448, 69.9830246, -168.8496246, 169.9279175
9: -73.9911880, 73.2073898, -76.5220490, 75.6940460, -149.6852112, 149.7294312

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2258256, upper bound: 193.2258256
time: 4.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2258256, upper bound: 193.2258256
time: 5.02 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -97.4439850, 77.5774689, -83.2625961, 66.4386368, -163.8826141, 160.8400574
1: -82.2057724, 69.0985870, -70.0714340, 58.9548073, -141.1605530, 139.1699829
2: -107.6984329, 69.9073410, -92.0089645, 59.7530975, -167.4515076, 161.9163055
3: -114.3310394, 60.8173828, -97.6565933, 51.9690704, -166.3001099, 158.4739685
4: -104.5442429, 80.4317856, -89.1863480, 68.6474380, -173.1916809, 169.6181335
5: -93.3289337, 72.9450378, -79.7642593, 62.1667137, -155.4956512, 152.7092896
6: -89.7876282, 86.5414124, -76.6809464, 73.9151840, -163.7028046, 163.2223511
7: -98.0236359, 82.1067276, -83.5298080, 70.1483078, -168.1719360, 165.6365356
8: -118.4481506, 81.1242447, -101.3009491, 69.3623581, -187.8104858, 182.4252014
9: -88.8717041, 87.8922729, -75.8289719, 75.0147858, -163.8864899, 163.7212524

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2210310, upper bound: 193.2172229
time: 6.44 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2159516, upper bound: 193.2159516
time: 5.47 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -81.2378311, 64.8312988, -82.6786575, 65.9964752, -147.2342529, 147.5099182
1: -68.3837051, 57.5365982, -69.5295792, 58.5330467, -126.9167480, 127.0661697
2: -89.7731781, 58.3186913, -91.4361420, 59.3242874, -149.0974579, 149.7548218
3: -95.2794189, 50.7288666, -97.0741730, 51.5299339, -146.8092957, 147.8030243
4: -87.0173950, 66.9982758, -88.5501251, 68.1753769, -155.1927643, 155.5484009
5: -77.8213120, 60.6631432, -79.1940460, 61.6374016, -139.4586945, 139.8571930
6: -74.8220978, 72.1273575, -76.1422424, 73.3440933, -148.1661987, 148.2695618
7: -81.5043869, 68.4531937, -82.9286499, 69.6382446, -151.1425934, 151.3818359
8: -98.8666458, 67.7094650, -100.6930008, 68.8089676, -167.6755981, 168.4024506
9: -73.9911880, 73.2073898, -75.1964569, 74.4152985, -148.4064636, 148.4038391

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1494581, upper bound: 193.1726807
time: 6.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1494581, upper bound: 193.1726807
time: 6.12 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -97.4439850, 77.5774689, -81.9401321, 65.4105911, -162.8545837, 159.5176086
1: -82.2057724, 69.0985870, -68.9147415, 58.0178375, -140.2235870, 138.0133057
2: -107.6984329, 69.9073410, -90.6221771, 58.7992935, -166.4977112, 160.5295105
3: -114.3310394, 60.8173828, -96.2103271, 51.0794029, -165.4104462, 157.0277100
4: -104.5442429, 80.4317856, -87.7598190, 67.5732269, -172.1174622, 168.1916046
5: -93.3289337, 72.9450378, -78.4865723, 61.0888138, -154.4177399, 151.4316101
6: -89.7876282, 86.5414124, -75.4651337, 72.6918182, -162.4794464, 162.0065460
7: -98.0236359, 82.1067276, -82.1891022, 69.0203247, -167.0439453, 164.2957916
8: -118.4481506, 81.1242447, -99.8033295, 68.2066727, -186.6547852, 180.9275818
9: -88.8717041, 87.8922729, -74.5247116, 73.7574310, -162.6291351, 162.4169922

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1328410, upper bound: 193.1484960
time: 6.97 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 131

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1071697, upper bound: 193.1188173
time: 7.10 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0605810, upper bound: 193.0927426
time: 5.51 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -80.1147842, 63.9604836, -84.0246582, 67.0425644, -147.1573181, 147.9851379
1: -67.3933716, 56.7388573, -70.7055969, 59.4865685, -126.8799362, 127.4444427
2: -88.6061707, 57.5048943, -92.8490372, 60.2935104, -148.8996887, 150.3539124
3: -94.0656891, 49.9612045, -98.5477219, 52.4346275, -146.5003204, 148.5088806
4: -85.8040085, 66.0848007, -90.0032654, 69.2684631, -155.0724640, 156.0880585
5: -76.7352524, 59.7337570, -80.4947510, 62.7323189, -139.4675751, 140.2285156
6: -73.7880402, 71.0802765, -77.3808975, 74.5888672, -148.3769073, 148.4611664
7: -80.3616943, 67.4917908, -84.2931061, 70.7854767, -151.1471558, 151.7848969
8: -97.6066284, 66.7158966, -102.2184448, 69.9830246, -167.5896301, 168.9343414
9: -72.8675003, 72.1278229, -76.5220490, 75.6940460, -148.5615540, 148.6498718

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1726807, upper bound: 193.1494591
time: 6.14 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1726807, upper bound: 193.1494591
time: 6.40 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -80.1147842, 63.9604836, -82.6786575, 65.9964752, -146.1112366, 146.6391296
1: -67.3933716, 56.7388573, -69.5295792, 58.5330467, -125.9264145, 126.2684250
2: -88.6061707, 57.5048943, -91.4361420, 59.3242874, -147.9304504, 148.9410248
3: -94.0656891, 49.9612045, -97.0741730, 51.5299339, -145.5956116, 147.0353546
4: -85.8040085, 66.0848007, -88.5501251, 68.1753769, -153.9793701, 154.6349182
5: -76.7352524, 59.7337570, -79.1940460, 61.6374016, -138.3726501, 138.9277954
6: -73.7880402, 71.0802765, -76.1422424, 73.3440933, -147.1321411, 147.2224884
7: -80.3616943, 67.4917908, -82.9286499, 69.6382446, -149.9999237, 150.4204407
8: -97.6066284, 66.7158966, -100.6930008, 68.8089676, -166.4155884, 167.4088745
9: -72.8675003, 72.1278229, -75.1964569, 74.4152985, -147.2828064, 147.3242798

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1369549, upper bound: 193.1369549
time: 4.99 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1369549, upper bound: 193.1369549
time: 4.76 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -81.2378311, 64.8312988, -93.2157745, 74.2332306, -155.4710541, 158.0470276
1: -68.3837051, 57.5365982, -78.5291367, 65.9542618, -134.3379211, 136.0657349
2: -89.7731781, 58.3186913, -102.9597778, 66.8783493, -156.6515198, 161.2784424
3: -95.2794189, 50.7288666, -109.2818756, 58.1466141, -153.4259644, 160.0107269
4: -87.0173950, 66.9982758, -99.8193741, 76.8472061, -163.8645935, 166.8176575
5: -77.8213120, 60.6631432, -89.3522644, 69.7041397, -147.5254059, 150.0154114
6: -74.8220978, 72.1273575, -85.8189316, 82.7380447, -157.5601349, 157.9462891
7: -81.5043869, 68.4531937, -93.6724472, 78.5660324, -160.0703888, 162.1256409
8: -98.8666458, 67.7094650, -113.2140656, 77.4516983, -176.3183136, 180.9235229
9: -73.9911880, 73.2073898, -85.0225220, 83.9734421, -157.9645996, 158.2299194

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2469271, upper bound: 193.2328073
time: 7.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2469271, upper bound: 193.2328073
time: 6.31 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -97.4439850, 77.5774689, -92.4088364, 73.5937347, -171.0376892, 169.9862976
1: -82.2057724, 69.0985870, -77.8585892, 65.3917999, -147.5975494, 146.9571533
2: -107.6984329, 69.9073410, -102.0704117, 66.3052139, -174.0036011, 171.9777374
3: -114.3310394, 60.8173828, -108.3391953, 57.6537971, -171.9848328, 169.1565857
4: -104.5442429, 80.4317856, -98.9548416, 76.1902084, -180.7344513, 179.3866272
5: -93.3289337, 72.9450378, -88.5791245, 69.1065292, -162.4354553, 161.5241699
6: -89.7876282, 86.5414124, -85.0792007, 82.0256195, -171.8132477, 171.6206055
7: -98.0236359, 82.1067276, -92.8648605, 77.8920822, -175.9157104, 174.9715729
8: -118.4481506, 81.1242447, -112.2440414, 76.7941360, -195.2422485, 193.3682861
9: -88.8717041, 87.8922729, -84.2897110, 83.2541962, -172.1259003, 172.1819763

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2416816, upper bound: 193.2238069
time: 7.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2373979, upper bound: 193.2221415
time: 6.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -81.2378311, 64.8312988, -95.7926102, 76.2786789, -157.5164642, 160.6239014
1: -68.3837051, 57.5365982, -80.6553192, 67.7581940, -136.1418915, 138.1919250
2: -89.7731781, 58.3186913, -105.8632736, 68.7125015, -158.4856873, 164.1819458
3: -95.2794189, 50.7288666, -112.3945236, 59.6664543, -154.9458160, 163.1233826
4: -87.0173950, 66.9982758, -102.5667572, 78.9613342, -165.9787292, 169.5650330
5: -77.8213120, 60.6631432, -91.8319016, 71.5602570, -149.3815308, 152.4950409
6: -74.8220978, 72.1273575, -88.1777344, 84.9729919, -159.7950592, 160.3050842
7: -81.5043869, 68.4531937, -96.2714691, 80.7301788, -162.2345428, 164.7246704
8: -98.8666458, 67.7094650, -116.4032059, 79.4741821, -178.3408051, 184.1126709
9: -73.9911880, 73.2073898, -87.3099365, 86.2237701, -160.2149048, 160.5173340

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2329036, upper bound: 193.2191599
time: 7.27 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2329036, upper bound: 193.2191599
time: 7.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -97.4439850, 77.5774689, -95.0059738, 75.6548309, -173.0988007, 172.5834351
1: -82.2057724, 69.0985870, -80.0015640, 67.2098160, -149.4155579, 149.1001434
2: -107.6984329, 69.9073410, -104.9950714, 68.1532593, -175.8516846, 174.9024048
3: -114.3310394, 60.8173828, -111.4754791, 59.1860962, -173.5171356, 172.2928162
4: -104.5442429, 80.4317856, -101.7238312, 78.3208160, -182.8650513, 182.1556091
5: -93.3289337, 72.9450378, -91.0779877, 70.9783707, -164.3073120, 164.0230255
6: -89.7876282, 86.5414124, -87.4563446, 84.2786102, -174.0662384, 173.9977570
7: -98.0236359, 82.1067276, -95.4841919, 80.0731125, -178.0967407, 177.5908813
8: -118.4481506, 81.1242447, -115.4563217, 78.8324738, -197.2805634, 196.5805664
9: -88.8717041, 87.8922729, -86.5962601, 85.5222473, -174.3939514, 174.4885254

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2214129, upper bound: 193.2056543
time: 7.16 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2018853, upper bound: 193.1935035
time: 7.45 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -80.1147842, 63.9604836, -93.2157745, 74.2332306, -154.3480225, 157.1762390
1: -67.3933716, 56.7388573, -78.5291367, 65.9542618, -133.3475952, 135.2679749
2: -88.6061707, 57.5048943, -102.9597778, 66.8783493, -155.4845276, 160.4646301
3: -94.0656891, 49.9612045, -109.2818756, 58.1466141, -152.2122803, 159.2430573
4: -85.8040085, 66.0848007, -99.8193741, 76.8472061, -162.6511841, 165.9041748
5: -76.7352524, 59.7337570, -89.3522644, 69.7041397, -146.4393921, 149.0860291
6: -73.7880402, 71.0802765, -85.8189316, 82.7380447, -156.5260773, 156.8992004
7: -80.3616943, 67.4917908, -93.6724472, 78.5660324, -158.9277039, 161.1642456
8: -97.6066284, 66.7158966, -113.2140656, 77.4516983, -175.0583191, 179.9299622
9: -72.8675003, 72.1278229, -85.0225220, 83.9734421, -156.8409424, 157.1503448

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1917980, upper bound: 193.1567605
time: 6.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1917980, upper bound: 193.1567605
time: 7.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 15.89 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.2258256, upper bound: 193.2258256
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.2258256, upper bound: 193.2258256
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.2210310, upper bound: 193.2172229
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.2159516, upper bound: 193.2159516
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.1494581, upper bound: 193.1726807
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.1494581, upper bound: 193.1726807
IS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.1071697, upper bound: 193.1188173
IS_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.0605810, upper bound: 193.0927426
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.1726807, upper bound: 193.1494591
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.1726807, upper bound: 193.1494591
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.1369549, upper bound: 193.1369549
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.1369549, upper bound: 193.1369549
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.2469271, upper bound: 193.2328073
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.2469271, upper bound: 193.2328073
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.2416816, upper bound: 193.2238069
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.2373979, upper bound: 193.2221415
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.2329036, upper bound: 193.2191599
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.2329036, upper bound: 193.2191599
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.2214129, upper bound: 193.2056543
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.2018853, upper bound: 193.1935035
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.1917980, upper bound: 193.1567605
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.89
Output dim: 2, lower bound: -193.1917980, upper bound: 193.1567605
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.1888702, upper bound: 193.1561272
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.2033785, upper bound: 193.1845182
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.1888702, upper bound: 193.1561272
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.1591126, upper bound: 193.1982809
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.1567605, upper bound: 193.1917980
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.1591126, upper bound: 193.1982809
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.1567605, upper bound: 193.1917980
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.1579065, upper bound: 193.1939902
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.1562345, upper bound: 193.1889877
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.1579065, upper bound: 193.1939902
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.1562345, upper bound: 193.1889877
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.2626132, upper bound: 193.2681971
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.2608773, upper bound: 193.2631528
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.2626132, upper bound: 193.2681971
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.2608773, upper bound: 193.2631528
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.2615750, upper bound: 193.2650646
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.2601496, upper bound: 193.2601483
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.2615750, upper bound: 193.2650646
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.89
Output dim: 2, lower bound: -193.2601496, upper bound: 193.2601483
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=194.859130859375
rel_dist={2: [-193.2889031662745, 193.28890316692934]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2673907, upper bound: 193.2609621
time: 8.70 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2857449, upper bound: 193.2857449
time: 7.05 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.87 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.87
Output dim: 2, lower bound: -193.2673907, upper bound: 193.2609621
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.87
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

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2063317, upper bound: 193.1899251
time: 7.42 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2047223, upper bound: 193.1894454
time: 7.91 seconds

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

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2609622, upper bound: 193.2673907
time: 7.32 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2609622, upper bound: 193.2857449
time: 6.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 15.16 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 15.16
Output dim: 2, lower bound: -193.2063317, upper bound: 193.1899251
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 15.16
Output dim: 2, lower bound: -193.2047223, upper bound: 193.1894454
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 15.16
Output dim: 2, lower bound: -193.2609622, upper bound: 193.2673907
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 15.16
Output dim: 2, lower bound: -193.2609622, upper bound: 193.2857449

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -91.0616760, 72.6025925, -90.3467255, 71.9803696, -163.0420532, 162.9492798
1: -76.6826172, 64.4346008, -76.0835724, 63.9334526, -140.6160736, 140.5181732
2: -100.6281509, 65.3385468, -99.7906113, 64.8229523, -165.4510956, 165.1291504
3: -106.7591019, 56.8005905, -105.9114380, 56.3684654, -163.1275635, 162.7120361
4: -97.5726166, 75.0812912, -96.7517853, 74.4699554, -172.0425568, 171.8330688
5: -87.3083038, 68.0207214, -86.5812912, 67.5343475, -154.8426056, 154.6020203
6: -83.8796844, 80.8764267, -83.1768494, 80.1855316, -164.0652161, 164.0532837
7: -91.4320068, 76.7481995, -90.7341385, 76.1330338, -167.5650177, 167.4823303
8: -110.7104263, 75.8192291, -109.7502441, 75.1301117, -185.8405457, 185.5694733
9: -83.0200348, 82.0559158, -82.3684998, 81.3875198, -164.4075623, 164.4243774

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2046459, upper bound: 193.1893511
time: 7.78 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2046459, upper bound: 193.1893511
time: 7.25 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -84.9166107, 67.7433167, -91.5876465, 72.9767456, -157.8933563, 159.3309631
1: -71.4922867, 60.1197433, -77.0869446, 64.7990341, -136.2913208, 137.2066956
2: -93.8546295, 60.9453583, -101.2328720, 65.6968689, -159.5514984, 162.1782227
3: -99.6206512, 52.9888458, -107.4688568, 57.0582657, -156.6788940, 160.4577026
4: -90.9589005, 70.0375290, -98.0708618, 75.4904251, -166.4493103, 168.1083984
5: -81.3705750, 63.4064674, -87.7719727, 68.3816757, -149.7522583, 151.1784058
6: -78.2182312, 75.4004517, -84.3111954, 81.2355728, -159.4537964, 159.7116394
7: -85.2451477, 71.5695801, -91.9826126, 77.1693192, -162.4144592, 163.5521851
8: -103.3178558, 70.7129364, -111.3490295, 76.0571213, -179.3749695, 182.0619659
9: -77.3701935, 76.5061646, -83.4237061, 82.4342194, -159.8044128, 159.9298706

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1769388, upper bound: 193.1687406
time: 8.15 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1659006, upper bound: 193.1475265
time: 7.65 seconds

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
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1899251, upper bound: 193.2063316
time: 6.78 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1894454, upper bound: 193.2047223
time: 6.90 seconds

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

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1899251, upper bound: 193.2721922
time: 7.38 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1894454, upper bound: 193.2700589
time: 6.81 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 15.51 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.51
Output dim: 2, lower bound: -193.2046459, upper bound: 193.1893511
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.51
Output dim: 2, lower bound: -193.2046459, upper bound: 193.1893511
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.51
Output dim: 2, lower bound: -193.1769388, upper bound: 193.1687406
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.51
Output dim: 2, lower bound: -193.1659006, upper bound: 193.1475265
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.51
Output dim: 2, lower bound: -193.1899251, upper bound: 193.2063316
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.51
Output dim: 2, lower bound: -193.1894454, upper bound: 193.2047223
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.51
Output dim: 2, lower bound: -193.1899251, upper bound: 193.2721922
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.51
Output dim: 2, lower bound: -193.1894454, upper bound: 193.2700589

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -84.0246582, 67.0425644, -90.3467255, 71.9803696, -156.0050354, 157.3892517
1: -70.7055969, 59.4865685, -76.0835724, 63.9334526, -134.6390533, 135.5701294
2: -92.8490372, 60.2935104, -99.7906113, 64.8229523, -157.6719818, 160.0840912
3: -98.5477219, 52.4346275, -105.9114380, 56.3684654, -154.9161835, 158.3460541
4: -90.0032654, 69.2684631, -96.7517853, 74.4699554, -164.4732056, 166.0202484
5: -80.4947510, 62.7323189, -86.5812912, 67.5343475, -148.0290680, 149.3136139
6: -77.3808975, 74.5888672, -83.1768494, 80.1855316, -157.5664368, 157.7657166
7: -84.2931061, 70.7854767, -90.7341385, 76.1330338, -160.4261169, 161.5196075
8: -102.2184448, 69.9830246, -109.7502441, 75.1301117, -177.3485565, 179.7332611
9: -76.5220490, 75.6940460, -82.3684998, 81.3875198, -157.9095764, 158.0625305

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1969195, upper bound: 193.1865065
time: 7.19 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1969195, upper bound: 193.1899251
time: 7.12 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -82.6786575, 65.9964752, -90.3467255, 71.9803696, -154.6590271, 156.3431854
1: -69.5295792, 58.5330467, -76.0835724, 63.9334526, -133.4630127, 134.6165924
2: -91.4361420, 59.3242874, -99.7906113, 64.8229523, -156.2590942, 159.1148529
3: -97.0741730, 51.5299339, -105.9114380, 56.3684654, -153.4426422, 157.4413452
4: -88.5501251, 68.1753769, -96.7517853, 74.4699554, -163.0200500, 164.9271545
5: -79.1940460, 61.6374016, -86.5812912, 67.5343475, -146.7283783, 148.2186890
6: -76.1422424, 73.3440933, -83.1768494, 80.1855316, -156.3277588, 156.5209351
7: -82.9286499, 69.6382446, -90.7341385, 76.1330338, -159.0616760, 160.3723755
8: -100.6930008, 68.8089676, -109.7502441, 75.1301117, -175.8231201, 178.5592041
9: -75.1964569, 74.4152985, -82.3684998, 81.3875198, -156.5839691, 156.7837982

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1969195, upper bound: 193.1865065
time: 8.60 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1969195, upper bound: 193.1899251
time: 7.24 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -82.1030807, 65.5110016, -90.8689880, 72.4062653, -154.5093384, 156.3799896
1: -69.1484985, 58.1506844, -76.4883575, 64.2961502, -133.4446411, 134.6390381
2: -90.7494965, 58.9517021, -100.4390259, 65.1873779, -155.9368439, 159.3907318
3: -96.3208389, 51.2665062, -106.6261749, 56.6183777, -152.9391785, 157.8926544
4: -87.9447784, 67.7455063, -97.3007202, 74.9047775, -162.8495483, 165.0461884
5: -78.6721344, 61.3176727, -87.0826645, 67.8486862, -146.5207825, 148.4003143
6: -75.6343994, 72.9155807, -83.6511765, 80.6009521, -156.2353516, 156.5667267
7: -82.4298553, 69.2151566, -91.2631760, 76.5679245, -158.9977722, 160.4783325
8: -99.9339523, 68.4173889, -110.4835129, 75.4703674, -175.4043121, 178.9009094
9: -74.8148880, 73.9961014, -82.7712326, 81.7927628, -156.6076202, 156.7673187

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1405829, upper bound: 193.1490184
time: 5.65 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1405829, upper bound: 193.1687397
time: 6.67 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -97.6012573, 77.7067261, -89.5936356, 71.3935471, -168.9948120, 167.3003540
1: -82.3636475, 69.2226181, -75.4251633, 63.4071693, -145.7707977, 144.6477661
2: -107.8962708, 70.0329819, -99.0301437, 64.2793808, -172.1756592, 169.0631256
3: -114.5543060, 60.9034615, -105.1359558, 55.8406258, -170.3949280, 166.0394135
4: -104.6908112, 80.5741272, -95.9344940, 73.8652878, -178.5560913, 176.5086212
5: -93.4818344, 73.0631027, -85.8589020, 66.9028473, -160.3846741, 158.9219818
6: -89.9388351, 86.6922684, -82.4805222, 79.4731140, -169.4119568, 169.1727905
7: -98.2259903, 82.2728729, -89.9846802, 75.4994049, -173.7253876, 172.2575378
8: -118.6706924, 81.2304306, -108.9427567, 74.4300766, -193.1007690, 190.1731873
9: -89.0468750, 88.0241699, -81.6110535, 80.6564102, -169.7032776, 169.6352234

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1369475, upper bound: 193.1369475
time: 5.20 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1369475, upper bound: 193.1475265
time: 5.46 seconds

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

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1893511, upper bound: 193.2046459
time: 6.87 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1893511, upper bound: 193.2046459
time: 6.90 seconds

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

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1687406, upper bound: 193.1769388
time: 10.08 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1475264, upper bound: 193.1659006
time: 7.09 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -93.2157745, 74.2332306, -101.0141296, 80.3891296, -173.6049042, 175.2473602
1: -78.5291367, 65.9542618, -85.1496429, 71.4373550, -149.9664612, 151.1038666
2: -102.9597778, 66.8783493, -111.5755234, 72.4665909, -175.4263611, 178.4538727
3: -109.2818756, 58.1466141, -118.3838196, 62.9882164, -172.2700653, 176.5304108
4: -99.8193741, 76.8472061, -108.2024384, 83.2869110, -183.1062775, 185.0496368
5: -89.3522644, 69.7041397, -96.9042664, 75.5704041, -164.9226685, 166.6083679
6: -85.8189316, 82.7380447, -93.0159988, 89.7023239, -175.5212555, 175.7540436
7: -93.6724472, 78.5660324, -101.5856857, 85.1714630, -178.8439026, 180.1517181
8: -113.2140656, 77.4516983, -122.6185379, 83.9093781, -197.1234436, 200.0702362
9: -85.0225220, 83.9734421, -92.2251968, 91.0230179, -176.0455322, 176.1986389

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2700564, upper bound: 193.2700589
time: 6.21 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2700564, upper bound: 193.2700589
time: 5.48 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -95.7926102, 76.2786789, -95.7018204, 76.1830826, -171.9756775, 171.9804993
1: -80.6553192, 67.7581940, -80.6637650, 67.7082748, -148.3635406, 148.4219513
2: -105.8632736, 68.7125015, -105.7180862, 68.6711731, -174.5344543, 174.4305878
3: -112.3945236, 59.6664543, -112.2145920, 59.6933365, -172.0878448, 171.8810425
4: -102.5667572, 78.9613342, -102.4821167, 78.9243774, -181.4911346, 181.4434509
5: -91.8319016, 71.5602570, -91.7754211, 71.5902252, -163.4221039, 163.3356781
6: -88.1777344, 84.9729919, -88.1203842, 84.9695511, -173.1472778, 173.0933380
7: -96.2714691, 80.7301788, -96.2466431, 80.7034836, -176.9749298, 176.9768219
8: -116.4032059, 79.4741821, -116.2219849, 79.4856949, -195.8889008, 195.6961670
9: -87.3099365, 86.2237701, -87.3521957, 86.2275620, -173.5375061, 173.5759583

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2631647, upper bound: 193.2608490
time: 6.60 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2600944, upper bound: 193.2600933
time: 5.27 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 13.18 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 2, lower bound: -193.1969195, upper bound: 193.1865065
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 2, lower bound: -193.1969195, upper bound: 193.1899251
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 2, lower bound: -193.1969195, upper bound: 193.1865065
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 2, lower bound: -193.1969195, upper bound: 193.1899251
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 13.18
Output dim: 2, lower bound: -193.1405829, upper bound: 193.1490184
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 2, lower bound: -193.1405829, upper bound: 193.1687397
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 13.18
Output dim: 2, lower bound: -193.1369475, upper bound: 193.1369475
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 13.18
Output dim: 2, lower bound: -193.1369475, upper bound: 193.1475265
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 2, lower bound: -193.1893511, upper bound: 193.2046459
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 2, lower bound: -193.1893511, upper bound: 193.2046459
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 2, lower bound: -193.1687406, upper bound: 193.1769388
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 2, lower bound: -193.1475264, upper bound: 193.1659006
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 2, lower bound: -193.2700564, upper bound: 193.2700589
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 2, lower bound: -193.2700564, upper bound: 193.2700589
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 2, lower bound: -193.2631647, upper bound: 193.2608490
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 2, lower bound: -193.2600944, upper bound: 193.2600933

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -84.0246582, 67.0425644, -83.9879532, 67.0126495, -151.0373077, 151.0305176
1: -70.7055969, 59.4865685, -70.6738205, 59.4610100, -130.1666107, 130.1603699
2: -92.8490372, 60.2935104, -92.8083878, 60.2672119, -153.1162415, 153.1018982
3: -98.5477219, 52.4346275, -98.5040970, 52.4125519, -150.9602356, 150.9387207
4: -90.0032654, 69.2684631, -89.9642639, 69.2382050, -159.2414703, 159.2327271
5: -80.4947510, 62.7323189, -80.4584503, 62.7057686, -143.2005157, 143.1907654
6: -77.3808975, 74.5888672, -77.3471832, 74.5560226, -151.9369202, 151.9360504
7: -84.2931061, 70.7854767, -84.2560425, 70.7542038, -155.0472717, 155.0415192
8: -102.2184448, 69.9830246, -102.1743546, 69.9514542, -172.1698914, 172.1573639
9: -76.5220490, 75.6940460, -76.4876862, 75.6604156, -152.1824646, 152.1817322

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2275952, upper bound: 193.2322064
time: 6.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2257897, upper bound: 193.2257897
time: 5.38 seconds

## BFS IS instance: IS_A1_B1_A1_B2

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

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2275952, upper bound: 193.2379488
time: 6.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2257897, upper bound: 193.2298341
time: 5.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -82.6786575, 65.9964752, -83.9879532, 67.0126495, -149.6913147, 149.9844208
1: -69.5295792, 58.5330467, -70.6738205, 59.4610100, -128.9905853, 129.2068329
2: -91.4361420, 59.3242874, -92.8083878, 60.2672119, -151.7033539, 152.1326599
3: -97.0741730, 51.5299339, -98.5040970, 52.4125519, -149.4867096, 150.0340271
4: -88.5501251, 68.1753769, -89.9642639, 69.2382050, -157.7883301, 158.1396484
5: -79.1940460, 61.6374016, -80.4584503, 62.7057686, -141.8998108, 142.0958557
6: -76.1422424, 73.3440933, -77.3471832, 74.5560226, -150.6982422, 150.6912842
7: -82.9286499, 69.6382446, -84.2560425, 70.7542038, -153.6828308, 153.8942871
8: -100.6930008, 68.8089676, -102.1743546, 69.9514542, -170.6444550, 170.9833221
9: -75.1964569, 74.4152985, -76.4876862, 75.6604156, -150.8568726, 150.9029846

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1657318, upper bound: 193.1624546
time: 8.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1572184, upper bound: 193.1437463
time: 8.34 seconds

## BFS IS instance: IS_A1_B1_A2_B2

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

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1657318, upper bound: 193.1696800
time: 7.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1572184, upper bound: 193.1478513
time: 8.99 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -82.1030807, 65.5110016, -94.9637680, 75.6227875, -157.7258606, 160.4747620
1: -69.1484985, 58.1506844, -79.9701691, 67.1814423, -136.3299255, 138.1208496
2: -90.7494965, 58.9517021, -104.9505005, 68.1225052, -158.8719788, 163.9022064
3: -96.3208389, 51.2665062, -111.4274902, 59.1586952, -155.4795227, 162.6939850
4: -87.9447784, 67.7455063, -101.6781387, 78.2871017, -166.2318726, 169.4236450
5: -78.6721344, 61.3176727, -91.0384827, 70.9506989, -149.6228333, 152.3561249
6: -75.6343994, 72.9155807, -87.4187622, 84.2443848, -159.8787842, 160.3343506
7: -82.4298553, 69.2151566, -95.4447403, 80.0424271, -162.4722900, 164.6598969
8: -99.9339523, 68.4173889, -115.4128799, 78.7981949, -178.7321472, 183.8302612
9: -74.8148880, 73.9961014, -86.5639114, 85.4846268, -160.2995148, 160.5600128

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 131

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0524005, upper bound: 193.1344639
time: 6.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0493352, upper bound: 193.1098262
time: 6.02 seconds

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

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1491168, upper bound: 193.1706457
time: 6.92 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1478517, upper bound: 193.1674226
time: 7.68 seconds

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

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1491168, upper bound: 193.1706457
time: 6.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1478517, upper bound: 193.1674226
time: 8.82 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -95.0455170, 75.6861496, -82.1030807, 65.5110016, -160.5565186, 157.7892303
1: -80.0345383, 67.2360916, -69.1484985, 58.1506844, -138.1852264, 136.3845673
2: -105.0386887, 68.1823349, -90.7494965, 58.9517021, -163.9903870, 158.9318237
3: -111.5194550, 59.2090759, -96.3208389, 51.2665062, -162.7859344, 155.5298920
4: -101.7661514, 78.3528442, -87.9447784, 67.7455063, -169.5116577, 166.2976227
5: -91.1158981, 71.0075302, -78.6721344, 61.3176727, -152.4335327, 149.6796570
6: -87.4921494, 84.3141098, -75.6343994, 72.9155807, -160.4077301, 159.9485168
7: -95.5242920, 80.1063766, -82.4298553, 69.2151566, -164.7394409, 162.5362244
8: -115.5053482, 78.8642349, -99.9339523, 68.4173889, -183.9227295, 178.7981873
9: -86.6329727, 85.5568161, -74.8148880, 73.9961014, -160.6290741, 160.3716888

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1114320, upper bound: 193.1201114
time: 6.99 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0979964, upper bound: 193.1035628
time: 6.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -93.7440338, 74.6534500, -97.6012573, 77.7067261, -171.4507599, 172.2546844
1: -78.9517670, 66.3300247, -82.3636475, 69.2226181, -148.1743317, 148.6936188
2: -103.6020584, 67.2550430, -107.8962708, 70.0329819, -173.6350250, 175.1513062
3: -110.0003128, 58.4151230, -114.5543060, 60.9034615, -170.9037628, 172.9694214
4: -100.3718948, 77.2924805, -104.6908112, 80.5741272, -180.9460144, 181.9832916
5: -89.8678513, 70.0445709, -93.4818344, 73.0631027, -162.9309540, 163.5263977
6: -86.2985229, 83.1645126, -89.9388351, 86.6922684, -172.9907837, 173.1033478
7: -94.2209702, 79.0183792, -98.2259903, 82.2728729, -176.4938354, 177.2443695
8: -113.9362717, 77.8026962, -118.6706924, 81.2304306, -195.1667023, 196.4733887
9: -85.4518967, 84.3974991, -89.0468750, 88.0241699, -173.4760742, 173.4443665

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 131

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0595169, upper bound: 193.0879675
time: 5.86 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0597141, upper bound: 193.0884112
time: 7.00 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -93.2157745, 74.2332306, -93.2157745, 74.2332306, -167.4490051, 167.4490051
1: -78.5291367, 65.9542618, -78.5291367, 65.9542618, -144.4833679, 144.4833679
2: -102.9597778, 66.8783493, -102.9597778, 66.8783493, -169.8381195, 169.8381195
3: -109.2818756, 58.1466141, -109.2818756, 58.1466141, -167.4284668, 167.4284668
4: -99.8193741, 76.8472061, -99.8193741, 76.8472061, -176.6665802, 176.6665802
5: -89.3522644, 69.7041397, -89.3522644, 69.7041397, -159.0563965, 159.0563965
6: -85.8189316, 82.7380447, -85.8189316, 82.7380447, -168.5569763, 168.5569763
7: -93.6724472, 78.5660324, -93.6724472, 78.5660324, -172.2384796, 172.2384796
8: -113.2140656, 77.4516983, -113.2140656, 77.4516983, -190.6657715, 190.6657715
9: -85.0225220, 83.9734421, -85.0225220, 83.9734421, -168.9959717, 168.9959717

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2613859, upper bound: 193.2648301
time: 6.34 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2604632, upper bound: 193.2619343
time: 6.25 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -93.2157745, 74.2332306, -95.7926102, 76.2786789, -169.4944305, 170.0258484
1: -78.5291367, 65.9542618, -80.6553192, 67.7581940, -146.2873230, 146.6095428
2: -102.9597778, 66.8783493, -105.8632736, 68.7125015, -171.6722717, 172.7416229
3: -109.2818756, 58.1466141, -112.3945236, 59.6664543, -168.9483185, 170.5411072
4: -99.8193741, 76.8472061, -102.5667572, 78.9613342, -178.7807007, 179.4139557
5: -89.3522644, 69.7041397, -91.8319016, 71.5602570, -160.9125214, 161.5360260
6: -85.8189316, 82.7380447, -88.1777344, 84.9729919, -170.7919159, 170.9157715
7: -93.6724472, 78.5660324, -96.2714691, 80.7301788, -174.4026184, 174.8374939
8: -113.2140656, 77.4516983, -116.4032059, 79.4741821, -192.6882477, 193.8549042
9: -85.0225220, 83.9734421, -87.3099365, 86.2237701, -171.2462921, 171.2833862

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2613859, upper bound: 193.2648301
time: 6.36 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2604632, upper bound: 193.2619343
time: 6.19 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -95.0455170, 75.6861496, -93.0567245, 74.0845871, -169.1300964, 168.7428741
1: -80.0345383, 67.2360916, -78.4630432, 65.8578720, -145.8924103, 145.6991272
2: -105.0386887, 68.1823349, -102.7993774, 66.7952347, -171.8338928, 170.9817200
3: -111.5194550, 59.2090759, -109.1148071, 58.0716476, -169.5910950, 168.3238525
4: -101.7661514, 78.3528442, -99.6463318, 76.7719879, -178.5381470, 177.9991455
5: -91.1158981, 71.0075302, -89.2398453, 69.6298065, -160.7456970, 160.2473755
6: -87.4921494, 84.3141098, -85.6930084, 82.6339264, -170.1260681, 170.0071106
7: -95.5242920, 80.1063766, -93.6014023, 78.4903336, -174.0145874, 173.7077789
8: -115.5053482, 78.8642349, -113.0443192, 77.3248367, -192.8301849, 191.9085388
9: -86.6329727, 85.5568161, -84.9501266, 83.8654175, -170.4983826, 170.5069275

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2426789, upper bound: 193.2430173
time: 7.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2374768, upper bound: 193.2324310
time: 7.45 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -93.7440338, 74.6534500, -107.6661682, 85.5774231, -179.3214569, 182.3195801
1: -78.9517670, 66.3300247, -90.9146881, 76.2742996, -155.2260590, 157.2446899
2: -103.6020584, 67.2550430, -118.9504166, 77.2294540, -180.8314972, 186.2054596
3: -110.0003128, 58.4151230, -126.2880554, 67.1716919, -177.1719971, 184.7031860
4: -100.3718948, 77.2924805, -115.4745941, 88.8733063, -189.2452087, 192.7670746
5: -89.8678513, 70.0445709, -103.2027359, 80.6833496, -170.5512085, 173.2472992
6: -86.2985229, 83.1645126, -99.1796646, 95.6149139, -181.9134369, 182.3441772
7: -94.2209702, 79.0183792, -108.4708710, 90.7691193, -184.9900818, 187.4892578
8: -113.9362717, 77.8026962, -130.6954041, 89.4215012, -203.3577728, 208.4981079
9: -85.4518967, 84.3974991, -98.3373337, 97.0876541, -182.5395508, 182.7348328

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2328413, upper bound: 193.2387106
time: 6.04 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2307014, upper bound: 193.2306962
time: 5.63 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 12.99 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.2275952, upper bound: 193.2322064
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.2257897, upper bound: 193.2257897
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.2275952, upper bound: 193.2379488
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.2257897, upper bound: 193.2298341
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.1657318, upper bound: 193.1624546
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.1572184, upper bound: 193.1437463
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.1657318, upper bound: 193.1696800
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.1572184, upper bound: 193.1478513
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.0524005, upper bound: 193.1344639
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.0493352, upper bound: 193.1098262
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.1491168, upper bound: 193.1706457
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.1478517, upper bound: 193.1674226
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.1491168, upper bound: 193.1706457
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.1478517, upper bound: 193.1674226
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.1114320, upper bound: 193.1201114
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.0979964, upper bound: 193.1035628
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.0595169, upper bound: 193.0879675
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.0597141, upper bound: 193.0884112
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.2613859, upper bound: 193.2648301
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.2604632, upper bound: 193.2619343
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.2613859, upper bound: 193.2648301
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.2604632, upper bound: 193.2619343
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.2426789, upper bound: 193.2430173
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.2374768, upper bound: 193.2324310
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.2328413, upper bound: 193.2387106
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 2, lower bound: -193.2307014, upper bound: 193.2306962

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -81.2378311, 64.8312988, -83.1712112, 66.3648453, -147.6026459, 148.0024567
1: -68.3837051, 57.5365982, -69.9933090, 58.8893356, -127.2730255, 127.5298996
2: -89.7731781, 58.3186913, -91.9067001, 59.6887665, -149.4619446, 150.2253723
3: -95.2794189, 50.7288666, -97.5458527, 51.9121170, -147.1914825, 148.2746887
4: -87.0173950, 66.9982758, -89.0888138, 68.5724640, -155.5898590, 156.0870972
5: -77.8213120, 60.6631432, -79.6747360, 62.0994682, -139.9207458, 140.3378754
6: -74.8220978, 72.1273575, -76.5960541, 73.8343353, -148.6564331, 148.7234039
7: -81.5043869, 68.4531937, -83.4383392, 70.0707779, -151.5751343, 151.8915253
8: -98.8666458, 67.7094650, -101.1920547, 69.2851334, -168.1517334, 168.9015045
9: -73.9911880, 73.2073898, -75.7457886, 74.9312286, -148.9223938, 148.9531860

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2236678, upper bound: 193.2282877
time: 6.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2183936, upper bound: 193.2253033
time: 7.66 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -97.4439850, 77.5774689, -82.0058517, 65.4410858, -162.8850708, 159.5833130
1: -82.2057724, 69.0985870, -69.0241089, 58.0783386, -140.2840881, 138.1226807
2: -107.6984329, 69.9073410, -90.6229935, 58.8605804, -166.5590057, 160.5303345
3: -114.3310394, 60.8173828, -96.1869125, 51.2026520, -165.5336914, 157.0043030
4: -104.5442429, 80.4317856, -87.8399658, 67.6235886, -172.1678314, 168.2717590
5: -93.3289337, 72.9450378, -78.5584183, 61.2346764, -154.5636139, 151.5034485
6: -89.7876282, 86.5414124, -75.5282364, 72.8040314, -162.5916595, 162.0696411
7: -98.0236359, 82.1067276, -82.2709045, 69.0961838, -167.1198120, 164.3775787
8: -118.4481506, 81.1242447, -99.7869186, 68.3370209, -186.7851410, 180.9111633
9: -88.8717041, 87.8922729, -74.6848831, 73.8941193, -162.7657928, 162.5771484

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2190376, upper bound: 193.2165526
time: 6.99 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2159030, upper bound: 193.2159030
time: 6.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -81.2378311, 64.8312988, -92.4128418, 73.5966415, -154.8344421, 157.2441101
1: -68.3837051, 57.5365982, -77.8614273, 65.3930817, -133.7767639, 135.3980255
2: -89.7731781, 58.3186913, -102.0743942, 66.3089828, -156.0821381, 160.3930511
3: -95.2794189, 50.7288666, -108.3409958, 57.6548882, -152.9342957, 159.0698395
4: -87.0173950, 66.9982758, -98.9588776, 76.1932297, -163.2106323, 165.9571533
5: -77.8213120, 60.6631432, -88.5825806, 69.1090775, -146.9303741, 149.2457275
6: -74.8220978, 72.1273575, -85.0822601, 82.0294266, -156.8515015, 157.2096252
7: -81.5043869, 68.4531937, -92.8695602, 77.8952026, -159.3995819, 161.3227539
8: -98.8666458, 67.7094650, -112.2501297, 76.7963257, -175.6629486, 179.9595795
9: -73.9911880, 73.2073898, -84.2941360, 83.2565765, -157.2477417, 157.5015259

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2386851, upper bound: 193.2341054
time: 9.19 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2338747, upper bound: 193.2307711
time: 8.11 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -97.4439850, 77.5774689, -91.1963654, 72.6315918, -170.0755768, 168.7738342
1: -82.2057724, 69.0985870, -76.8498917, 64.5458908, -146.7516174, 145.9484711
2: -107.6984329, 69.9073410, -100.7320175, 65.4432068, -173.1416321, 170.6393585
3: -114.3310394, 60.8173828, -106.9233780, 56.9125328, -171.2435760, 167.7407532
4: -104.5442429, 80.4317856, -97.6548538, 75.2043839, -179.7486267, 178.0866241
5: -93.3289337, 72.9450378, -87.4167099, 68.2081757, -161.5370941, 160.3617554
6: -89.7876282, 86.5414124, -83.9670258, 80.9538727, -170.7415009, 170.5084381
7: -98.0236359, 82.1067276, -91.6504364, 76.8764954, -174.9001312, 173.7571411
8: -118.4481506, 81.1242447, -110.7841797, 75.8055038, -194.2536469, 191.9084167
9: -88.8717041, 87.8922729, -83.1865692, 82.1725998, -171.0442810, 171.0788422

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2322109, upper bound: 193.2203950
time: 7.06 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2297636, upper bound: 193.2194976
time: 8.32 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -80.1147842, 63.9604836, -83.1712112, 66.3648453, -146.4796295, 147.1316681
1: -67.3933716, 56.7388573, -69.9933090, 58.8893356, -126.2826920, 126.7321548
2: -88.6061707, 57.5048943, -91.9067001, 59.6887665, -148.2949371, 149.4115753
3: -94.0656891, 49.9612045, -97.5458527, 51.9121170, -145.9777985, 147.5070038
4: -85.8040085, 66.0848007, -89.0888138, 68.5724640, -154.3764648, 155.1736145
5: -76.7352524, 59.7337570, -79.6747360, 62.0994682, -138.8347168, 139.4084930
6: -73.7880402, 71.0802765, -76.5960541, 73.8343353, -147.6223602, 147.6763306
7: -80.3616943, 67.4917908, -83.4383392, 70.0707779, -150.4324493, 150.9301300
8: -97.6066284, 66.7158966, -101.1920547, 69.2851334, -166.8917542, 167.9079285
9: -72.8675003, 72.1278229, -75.7457886, 74.9312286, -147.7987213, 147.8736115

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1059279, upper bound: 193.1025027
time: 8.99 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0984824, upper bound: 193.0967839
time: 8.16 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -94.4110184, 75.2102890, -82.0058517, 65.4410858, -159.8520966, 157.2161102
1: -79.5920258, 66.9631729, -69.0241089, 58.0783386, -137.6703644, 135.9872742
2: -104.4252853, 67.7291794, -90.6229935, 58.8605804, -163.2858582, 158.3521729
3: -110.8895798, 58.8527374, -96.1869125, 51.2026520, -162.0922241, 155.0396423
4: -101.2648163, 77.9255295, -87.8399658, 67.6235886, -168.8883972, 165.7654877
5: -90.3847275, 70.5737305, -78.5584183, 61.2346764, -151.6194000, 149.1321411
6: -86.9876404, 83.7866745, -75.5282364, 72.8040314, -159.7916718, 159.3149109
7: -94.9386749, 79.5337372, -82.2709045, 69.0961838, -164.0348511, 161.8045959
8: -114.8971329, 78.5315018, -99.7869186, 68.3370209, -183.2341461, 178.3184204
9: -85.9945374, 85.0681229, -74.6848831, 73.8941193, -159.8885956, 159.7530060

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 131

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0765884, upper bound: 193.0549345
time: 6.20 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0688726, upper bound: 193.0519266
time: 7.09 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -80.1147842, 63.9604836, -92.4128418, 73.5966415, -153.7114258, 156.3733063
1: -67.3933716, 56.7388573, -77.8614273, 65.3930817, -132.7864227, 134.6002808
2: -88.6061707, 57.5048943, -102.0743942, 66.3089828, -154.9151459, 159.5792542
3: -94.0656891, 49.9612045, -108.3409958, 57.6548882, -151.7205811, 158.3021545
4: -85.8040085, 66.0848007, -98.9588776, 76.1932297, -161.9972229, 165.0436707
5: -76.7352524, 59.7337570, -88.5825806, 69.1090775, -145.8443298, 148.3163452
6: -73.7880402, 71.0802765, -85.0822601, 82.0294266, -155.8174591, 156.1625366
7: -80.3616943, 67.4917908, -92.8695602, 77.8952026, -158.2568970, 160.3613586
8: -97.6066284, 66.7158966, -112.2501297, 76.7963257, -174.4029541, 178.9660034
9: -72.8675003, 72.1278229, -84.2941360, 83.2565765, -156.1240845, 156.4219513

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1224631, upper bound: 193.1133664
time: 7.14 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1117281, upper bound: 193.1048907
time: 7.57 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -94.4110184, 75.2102890, -91.1963654, 72.6315918, -167.0426025, 166.4066467
1: -79.5920258, 66.9631729, -76.8498917, 64.5458908, -144.1378937, 143.8130646
2: -104.4252853, 67.7291794, -100.7320175, 65.4432068, -169.8684998, 168.4611969
3: -110.8895798, 58.8527374, -106.9233780, 56.9125328, -167.8021088, 165.7761230
4: -101.2648163, 77.9255295, -97.6548538, 75.2043839, -176.4691925, 175.5803528
5: -90.3847275, 70.5737305, -87.4167099, 68.2081757, -158.5928955, 157.9904327
6: -86.9876404, 83.7866745, -83.9670258, 80.9538727, -167.9415131, 167.7536926
7: -94.9386749, 79.5337372, -91.6504364, 76.8764954, -171.8151703, 171.1841583
8: -114.8971329, 78.5315018, -110.7841797, 75.8055038, -190.7026367, 189.3156738
9: -85.9945374, 85.0681229, -83.1865692, 82.1725998, -168.1670990, 168.2546997

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 131

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0894530, upper bound: 193.0600894
time: 7.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0889618, upper bound: 193.0598702
time: 7.21 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -90.5806274, 72.1423569, -83.2078400, 66.3947144, -156.9753418, 155.3501892
1: -76.3360977, 64.1110611, -70.0250168, 58.9148331, -135.2509308, 134.1360779
2: -100.0519409, 65.0091476, -91.9472580, 59.7150116, -159.7669525, 156.9564056
3: -106.1935959, 56.5310936, -97.5893707, 51.9341583, -158.1277466, 154.1204681
4: -96.9940186, 74.7031021, -89.1277466, 68.6026764, -165.5966949, 163.8308411
5: -86.8256149, 67.7505493, -79.7109833, 62.1259613, -148.9515686, 147.4615326
6: -83.4009171, 80.4108963, -76.6297150, 73.8671112, -157.2680359, 157.0406189
7: -91.0368805, 76.3607178, -83.4753113, 70.1020355, -161.1389008, 159.8360138
8: -110.0484314, 75.2992630, -101.2360611, 69.3166656, -179.3650970, 176.5352936
9: -82.6289673, 81.6200256, -75.7800903, 74.9647980, -157.5937653, 157.4001160

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2259258, upper bound: 193.2361922
time: 7.94 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2210209, upper bound: 193.2341799
time: 6.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -105.9857101, 84.2554626, -82.0425339, 65.4709778, -171.4566956, 166.2980042
1: -89.4648972, 75.0955353, -69.0558548, 58.1038666, -147.5687561, 144.1513824
2: -117.0817261, 76.0124664, -90.6636124, 58.8868484, -175.9685669, 166.6760712
3: -124.3026657, 66.1267166, -96.2304764, 51.2247124, -175.5273590, 162.3571777
4: -113.6770248, 87.4657974, -87.8789215, 67.6538239, -181.3308411, 175.3447113
5: -101.5612717, 79.4161835, -78.5946960, 61.2611961, -162.8224640, 158.0108795
6: -97.6236725, 94.1071777, -75.5619202, 72.8368530, -170.4605255, 169.6690979
7: -106.7236023, 89.3189621, -82.3078995, 69.1274719, -175.8510437, 171.6268616
8: -128.6585846, 88.0551910, -99.8309708, 68.3685913, -197.0271606, 187.8861694
9: -96.7585678, 95.5710907, -74.7192307, 73.9277039, -170.6862488, 170.2903137

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2233261, upper bound: 193.2304952
time: 7.16 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2194976, upper bound: 193.2297636
time: 6.36 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 14.85 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.2236678, upper bound: 193.2282877
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.2183936, upper bound: 193.2253033
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.2190376, upper bound: 193.2165526
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.2159030, upper bound: 193.2159030
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.2386851, upper bound: 193.2341054
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.2338747, upper bound: 193.2307711
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.2322109, upper bound: 193.2203950
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.2297636, upper bound: 193.2194976
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.1059279, upper bound: 193.1025027
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.0984824, upper bound: 193.0967839
IS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.0765884, upper bound: 193.0549345
IS_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.0688726, upper bound: 193.0519266
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.1224631, upper bound: 193.1133664
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.1117281, upper bound: 193.1048907
IS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.0894530, upper bound: 193.0600894
IS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.0889618, upper bound: 193.0598702
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.2259258, upper bound: 193.2361922
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.2210209, upper bound: 193.2341799
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.2233261, upper bound: 193.2304952
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.85
Output dim: 2, lower bound: -193.2194976, upper bound: 193.2297636
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.85
Output dim: 2, lower bound: -193.1491168, upper bound: 193.1706457
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.85
Output dim: 2, lower bound: -193.1478517, upper bound: 193.1674226
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.85
Output dim: 2, lower bound: -193.2613859, upper bound: 193.2648301
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.85
Output dim: 2, lower bound: -193.2604632, upper bound: 193.2619343
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.85
Output dim: 2, lower bound: -193.2613859, upper bound: 193.2648301
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.85
Output dim: 2, lower bound: -193.2604632, upper bound: 193.2619343
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.85
Output dim: 2, lower bound: -193.2426789, upper bound: 193.2430173
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.85
Output dim: 2, lower bound: -193.2374768, upper bound: 193.2324310
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.85
Output dim: 2, lower bound: -193.2328413, upper bound: 193.2387106
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.85
Output dim: 2, lower bound: -193.2307014, upper bound: 193.2306962
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=194.859130859375
rel_dist={2: [-193.2888173876961, 193.28881738769616]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2616161, upper bound: 193.2592657
time: 10.04 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2853007, upper bound: 193.2853007
time: 10.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 20.89 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 20.89
Output dim: 2, lower bound: -193.2616161, upper bound: 193.2592657
IS_A2, status: Status.UNKNOWN, split count: 1, time: 20.89
Output dim: 2, lower bound: -193.2853007, upper bound: 193.2853007

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -92.5710983, 73.7949295, -93.5688782, 74.5447769, -167.1158447, 167.3638000
1: -77.9647217, 65.4957733, -78.8385544, 66.2028427, -144.1675568, 144.3343201
2: -102.2964478, 66.4200058, -103.3815002, 67.1444778, -169.4409180, 169.8014984
3: -108.5205307, 57.7372856, -109.6564636, 58.3831749, -166.9037018, 167.3937225
4: -99.1955872, 76.3285904, -100.2388535, 77.1570740, -176.3526459, 176.5674286
5: -88.7699814, 69.1550369, -89.7283173, 69.9561310, -158.7261047, 158.8833618
6: -85.2733154, 82.2249146, -86.1801758, 83.1030197, -168.3763123, 168.4050446
7: -92.9629669, 78.0268402, -94.0111084, 78.8858109, -171.8487854, 172.0379028
8: -112.5315170, 77.0708542, -113.6727600, 77.8774948, -190.4089966, 190.7436066
9: -84.4136353, 83.4204712, -85.3666153, 84.3352585, -168.7488708, 168.7870789

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1901906, upper bound: 193.1843653
time: 10.34 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1897154, upper bound: 193.1841976
time: 10.95 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -102.6823044, 81.7061234, -103.9861908, 82.7320251, -185.4143219, 185.6923218
1: -86.5658798, 72.6101608, -87.6619949, 73.5231018, -160.0889893, 160.2721558
2: -113.4187164, 73.6618729, -114.8480759, 74.5906448, -188.0093689, 188.5099487
3: -120.3311996, 64.0238419, -121.8501434, 64.8276520, -185.1588440, 185.8739777
4: -109.9954300, 84.6645660, -111.3940201, 85.7288055, -195.7242126, 196.0585785
5: -98.5205078, 76.8254395, -99.7721252, 77.7996979, -176.3201599, 176.5975647
6: -94.5557632, 91.1921463, -95.7494507, 92.3434296, -186.8991547, 186.9415283
7: -103.2783661, 86.5848770, -104.5885925, 87.6765594, -190.9549103, 191.1734619
8: -124.6306076, 85.2908020, -126.1875229, 86.3559647, -210.9865570, 211.4783325
9: -93.7667542, 92.5312347, -94.9571991, 93.6972733, -187.4640198, 187.4884338

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2704893, upper bound: 193.2697866
time: 9.39 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2695610, upper bound: 193.2695610
time: 7.43 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 18.13 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 18.13
Output dim: 2, lower bound: -193.1901906, upper bound: 193.1843653
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 18.13
Output dim: 2, lower bound: -193.1897154, upper bound: 193.1841976
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 18.13
Output dim: 2, lower bound: -193.2704893, upper bound: 193.2697866
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 18.13
Output dim: 2, lower bound: -193.2695610, upper bound: 193.2695610

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -86.7210617, 69.1733704, -84.8535080, 67.6612320, -154.3822784, 154.0268860
1: -72.9956818, 61.3825340, -71.4355164, 60.0745239, -133.0701904, 132.8180542
2: -95.8298264, 62.2271805, -93.7470398, 60.8976974, -156.7275238, 155.9742126
3: -101.6937027, 54.1072159, -99.4863892, 52.9740028, -154.6676941, 153.5935974
4: -92.9038391, 71.4950867, -90.8644028, 69.9562836, -162.8601227, 162.3594818
5: -83.1051254, 64.7586136, -81.2887726, 63.4035454, -146.5086670, 146.0473938
6: -79.8709030, 76.9979401, -78.1324539, 75.3157501, -155.1866455, 155.1303711
7: -87.0288086, 73.0703735, -85.1678238, 71.5017242, -158.5305023, 158.2381897
8: -105.4725418, 72.2195129, -103.1558380, 70.6540070, -176.1265411, 175.3753510
9: -79.0117874, 78.1315079, -77.3194733, 76.4563217, -155.4680939, 155.4509888

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1534357, upper bound: 193.1511908
time: 9.96 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1480336, upper bound: 193.1410017
time: 8.04 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -79.3927536, 63.3701057, -85.3847046, 68.1044540, -147.4972076, 148.7548065
1: -66.8219452, 56.2353439, -71.8411713, 60.4436913, -127.2656250, 128.0764923
2: -87.7619247, 56.9942665, -94.4055481, 61.2673454, -149.0292664, 151.3997955
3: -93.1953735, 49.5682411, -100.2201233, 53.2257919, -146.4211426, 149.7883606
4: -85.0240173, 65.5049057, -91.4216690, 70.3994446, -155.4234467, 156.9265594
5: -76.0374908, 59.2581787, -81.8040314, 63.7136002, -139.7510986, 141.0622101
6: -73.1308517, 70.4795227, -78.6207886, 75.7345963, -148.8654327, 149.1003113
7: -79.6800995, 66.9098282, -85.6915359, 71.9393234, -151.6194153, 152.6013184
8: -96.6674652, 66.1299973, -103.9010773, 71.0110855, -167.6785583, 170.0310669
9: -72.2891922, 71.5229492, -77.7117081, 76.8601379, -149.1493225, 149.2346497

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1527004, upper bound: 193.1507017
time: 9.23 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1475625, upper bound: 193.1408972
time: 11.14 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -96.2177277, 76.6027679, -94.5867081, 75.3113708, -171.5290985, 171.1894836
1: -81.0778122, 68.0650330, -79.6817169, 66.9137497, -147.9915619, 147.7467194
2: -106.2765350, 69.0291748, -104.4622040, 67.8554382, -174.1319427, 173.4913483
3: -112.7854004, 60.0104713, -110.8788986, 58.9912910, -171.7766724, 170.8893738
4: -103.0468674, 79.3261185, -101.2892151, 77.9663086, -181.0131836, 180.6152954
5: -92.2588348, 71.9617004, -90.6688995, 70.7296677, -162.9884796, 162.6305847
6: -88.5894394, 85.4190216, -87.0739517, 83.9486389, -172.5380707, 172.4929810
7: -96.7187881, 81.1084061, -95.0504837, 79.7148895, -176.4336853, 176.1588745
8: -116.8342133, 79.9377365, -114.8510361, 78.5712738, -195.4054718, 194.7887726
9: -87.7944260, 86.6867218, -86.2755737, 85.2002258, -172.9946442, 172.9622955

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2609171, upper bound: 193.2612075
time: 8.84 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2606066, upper bound: 193.2599740
time: 9.69 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -90.6086731, 72.1517563, -96.9466476, 77.1862259, -167.7948914, 169.0983887
1: -76.3593292, 64.1328735, -81.6251068, 68.5658188, -144.9251404, 145.7579651
2: -100.0985107, 65.0319138, -107.1293182, 69.5343857, -169.6329041, 172.1612244
3: -106.2946548, 56.5375023, -113.7387695, 60.3772888, -166.6719360, 170.2762604
4: -97.0005875, 74.7386627, -103.8045654, 79.9025040, -176.9030914, 178.5432281
5: -86.8544388, 67.7721405, -92.9390640, 72.4227753, -159.2772064, 160.7111816
6: -83.4273453, 80.4327850, -89.2339935, 85.9918289, -169.4191589, 169.6667786
7: -91.1199646, 76.4155502, -97.4315491, 81.6964417, -172.8164062, 173.8471069
8: -110.0876846, 75.2509689, -117.7813950, 80.4160004, -190.5036926, 193.0323639
9: -82.6773911, 81.6310425, -88.3639297, 87.2555313, -169.9329224, 169.9949646

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2600667, upper bound: 193.2610240
time: 9.35 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2598225, upper bound: 193.2598225
time: 10.45 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.12 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.12
Output dim: 2, lower bound: -193.1534357, upper bound: 193.1511908
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 21.12
Output dim: 2, lower bound: -193.1480336, upper bound: 193.1410017
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.12
Output dim: 2, lower bound: -193.1527004, upper bound: 193.1507017
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 21.12
Output dim: 2, lower bound: -193.1475625, upper bound: 193.1408972
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.12
Output dim: 2, lower bound: -193.2609171, upper bound: 193.2612075
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.12
Output dim: 2, lower bound: -193.2606066, upper bound: 193.2599740
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.12
Output dim: 2, lower bound: -193.2600667, upper bound: 193.2610240
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.12
Output dim: 2, lower bound: -193.2598225, upper bound: 193.2598225

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -83.9088211, 66.9434509, -82.9366379, 66.1400452, -150.0488434, 149.8800964
1: -70.6530609, 59.4163017, -69.8385696, 58.7341766, -129.3871765, 129.2548523
2: -92.7270279, 60.2355194, -91.6317520, 59.5387077, -152.2657318, 151.8672791
3: -98.3971405, 52.3841095, -97.2401276, 51.7991791, -150.1963196, 149.6242371
4: -89.8875122, 69.2028275, -88.8081894, 68.3959961, -158.2835083, 158.0110168
5: -80.4062347, 62.6713867, -79.4492035, 61.9810829, -142.3873138, 142.1205597
6: -77.2872314, 74.5139465, -76.3730621, 73.6221008, -150.9093323, 150.8870087
7: -84.2148438, 70.7189255, -83.2502670, 69.8971939, -154.1120300, 153.9691925
8: -102.0923538, 69.9240189, -100.8519516, 69.0892715, -171.1816254, 170.7759705
9: -76.4595566, 75.6212463, -75.5787201, 74.7451172, -151.2046204, 151.1999664

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 158

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0604918, upper bound: 193.0584884
time: 8.93 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0508884, upper bound: 193.0504062
time: 8.53 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -76.6249390, 61.1727829, -83.6031113, 66.6891403, -143.3140717, 144.7758789
1: -64.5157852, 54.2989807, -70.3566360, 59.1969299, -123.7127151, 124.6556168
2: -84.7075195, 55.0313988, -92.4377365, 60.0032921, -144.7108154, 147.4690857
3: -89.9490814, 47.8749313, -98.1295624, 52.1351776, -142.0842438, 146.0044861
4: -82.0601120, 63.2497482, -89.5133972, 68.9469147, -151.0070190, 152.7631531
5: -73.3822784, 57.2026634, -80.0941620, 62.3917007, -135.7739563, 137.2967987
6: -70.5909958, 68.0357895, -76.9838867, 74.1613235, -144.7523041, 145.0196686
7: -76.9113693, 64.5927734, -83.9078903, 70.4472351, -147.3586121, 148.5006256
8: -93.3384247, 63.8707237, -101.7548904, 69.5563889, -162.8947906, 165.6256104
9: -69.7757111, 69.0542221, -76.0937729, 75.2699051, -145.0455933, 145.1479950

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0586159, upper bound: 193.0565423
time: 8.62 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0368368, upper bound: 193.0365601
time: 8.73 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -93.5646744, 74.4981384, -92.6399384, 73.7668610, -167.3315430, 167.1380768
1: -78.8702164, 66.2088776, -78.0620422, 65.5522308, -144.4224548, 144.2709198
2: -103.3490295, 67.1477585, -102.3142242, 66.4741516, -169.8231812, 169.4619751
3: -109.6755524, 58.3835487, -108.5972977, 57.7982674, -167.4738007, 166.9808502
4: -100.2022018, 77.1671600, -99.2021484, 76.3819504, -176.5841522, 176.3693085
5: -89.7154617, 69.9950180, -88.8023682, 69.2867508, -159.0021973, 158.7973938
6: -86.1539154, 83.0759125, -85.2880173, 82.2296143, -168.3835144, 168.3639221
7: -94.0650024, 78.8882980, -93.1035690, 78.0861435, -172.1511536, 171.9918671
8: -113.6471176, 77.7709351, -112.5125580, 76.9816055, -190.6287079, 190.2834930
9: -85.3842392, 84.3170090, -84.5078735, 83.4614716, -168.8457031, 168.8248901

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2359095, upper bound: 193.2356512
time: 10.57 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2339948, upper bound: 193.2341450
time: 9.21 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -108.8467331, 86.5147781, -91.1908798, 72.6177139, -181.4644470, 177.7056580
1: -91.8938751, 77.1050873, -76.8590622, 64.5458145, -156.4396973, 153.9641418
2: -120.2416687, 78.0635529, -100.7159805, 65.4421005, -185.6837769, 178.7795410
3: -127.6391220, 67.9027939, -106.9148483, 56.9154358, -184.5545349, 174.8176270
4: -116.7518616, 89.8270950, -97.6507874, 75.2056046, -191.9574280, 187.4778748
5: -104.3313141, 81.5671387, -87.4151077, 68.2142944, -172.5456085, 168.9822388
6: -100.2627869, 96.6606674, -83.9598999, 80.9490051, -181.2117920, 180.6205750
7: -109.6249619, 91.7414246, -91.6503983, 76.8737717, -186.4987030, 183.3917847
8: -132.1088104, 90.4240799, -110.7659912, 75.8039246, -207.9127197, 201.1900330
9: -99.3989487, 98.1557083, -83.1881866, 82.1728287, -181.5717773, 181.3439026

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2340705, upper bound: 193.2313044
time: 8.51 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2329533, upper bound: 193.2309662
time: 8.32 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -87.9850006, 70.0698547, -95.0749359, 75.7014923, -163.6864929, 165.1447449
1: -74.1765289, 62.2980270, -80.0695343, 67.2575760, -141.4340668, 142.3675537
2: -97.2030106, 63.1712494, -105.0630188, 68.2058640, -165.4088745, 168.2342682
3: -103.2210999, 54.9293442, -111.5459747, 59.2312050, -162.4522858, 166.4753113
4: -94.1884537, 72.6040115, -101.7984314, 78.3778992, -172.5663452, 174.4024353
5: -84.3388214, 65.8279114, -91.1448517, 71.0381699, -155.3769836, 156.9727325
6: -81.0201569, 78.1164856, -87.5160599, 84.3409042, -165.3610382, 165.6325378
7: -88.4971237, 74.2205200, -95.5594025, 80.1331406, -168.6302643, 169.7799072
8: -106.9355850, 73.1070786, -115.5314789, 78.8877182, -185.8233032, 188.6385498
9: -80.2961807, 79.2884979, -86.6678619, 85.5843964, -165.8805542, 165.9563599

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2352694, upper bound: 193.2354327
time: 10.13 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2309769, upper bound: 193.2330468
time: 7.31 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -102.2660904, 81.3098526, -93.4279633, 74.3942032, -176.6602936, 174.7378082
1: -86.3481979, 72.4815598, -78.6998291, 66.1120377, -152.4602051, 151.1813660
2: -112.9980545, 73.3698959, -103.2456818, 67.0288315, -180.0268860, 176.6155701
3: -120.0096970, 63.8229027, -109.6274261, 58.2272720, -178.2369385, 173.4503174
4: -109.6650925, 84.4345627, -100.0359039, 77.0384903, -186.7035828, 184.4704590
5: -97.9889984, 76.6317978, -89.5673752, 69.8197098, -167.8087158, 166.1991272
6: -94.2046967, 90.8053131, -86.0078354, 82.8854980, -177.0901947, 176.8131409
7: -103.0331955, 86.2221375, -93.9101562, 78.7545471, -181.7877502, 180.1322784
8: -124.1942139, 84.9336853, -113.5452118, 77.5453491, -201.7395477, 198.4788971
9: -93.3775177, 92.2128067, -85.1711426, 84.1195450, -177.4970551, 177.3839417

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 246
type: B, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2334656, upper bound: 193.2311551
time: 9.30 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2303621, upper bound: 193.2303621
time: 6.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 17.60 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 17.60
Output dim: 2, lower bound: -193.0604918, upper bound: 193.0584884
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 17.60
Output dim: 2, lower bound: -193.0508884, upper bound: 193.0504062
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 17.60
Output dim: 2, lower bound: -193.0586159, upper bound: 193.0565423
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 17.60
Output dim: 2, lower bound: -193.0368368, upper bound: 193.0365601
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.60
Output dim: 2, lower bound: -193.2359095, upper bound: 193.2356512
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.60
Output dim: 2, lower bound: -193.2339948, upper bound: 193.2341450
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.60
Output dim: 2, lower bound: -193.2340705, upper bound: 193.2313044
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.60
Output dim: 2, lower bound: -193.2329533, upper bound: 193.2309662
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.60
Output dim: 2, lower bound: -193.2352694, upper bound: 193.2354327
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.60
Output dim: 2, lower bound: -193.2309769, upper bound: 193.2330468
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.60
Output dim: 2, lower bound: -193.2334656, upper bound: 193.2311551
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.60
Output dim: 2, lower bound: -193.2303621, upper bound: 193.2303621

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -91.5741196, 72.8906708, -89.8379288, 71.5038834, -163.0780029, 162.7286072
1: -77.1738281, 64.7808304, -75.6738281, 63.5414085, -140.7152100, 140.4546509
2: -101.1271973, 65.7136917, -99.1865768, 64.4552536, -165.5824432, 164.9002533
3: -107.3037033, 57.1240387, -105.2583160, 56.0249672, -163.3286438, 162.3823547
4: -98.0385361, 75.5350418, -96.1567917, 74.0845642, -172.1231079, 171.6918335
5: -87.8112411, 68.4869919, -86.1219025, 67.1637497, -154.9749908, 154.6088867
6: -84.3120728, 81.2844772, -82.6952515, 79.7075653, -164.0196075, 163.9797363
7: -92.0646896, 77.2194443, -90.2874222, 75.7368851, -167.8015747, 167.5068665
8: -111.2073517, 76.0761719, -109.0781174, 74.5958023, -185.8031616, 185.1542816
9: -83.5723495, 82.5057068, -81.9566574, 80.9111023, -164.4834137, 164.4623718

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 131

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2168512, upper bound: 193.2166849
time: 9.95 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2165943, upper bound: 193.2161647
time: 9.97 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -90.1925049, 71.7686920, -94.3096008, 74.9903259, -165.1828308, 166.0782776
1: -75.9876556, 63.7836685, -79.3730164, 66.6307068, -142.6183624, 143.1566772
2: -99.5861435, 64.7197266, -104.0875168, 67.6035538, -167.1896820, 168.8072510
3: -105.6578674, 56.2478371, -110.5012283, 58.6820641, -164.3399353, 166.7490387
4: -96.5428696, 74.3962097, -100.9774017, 77.7602005, -174.3030701, 175.3735962
5: -86.4856567, 67.4368134, -90.4192123, 70.4379807, -156.9236145, 157.8560028
6: -83.0363846, 80.0366287, -86.8141098, 83.6313248, -166.6677094, 166.8507233
7: -90.6820602, 76.0591965, -94.8009491, 79.4732208, -170.1552277, 170.8601379
8: -109.5065613, 74.8885574, -114.4666977, 78.1353760, -187.6419373, 189.3552551
9: -82.3150024, 81.2464218, -86.0532532, 84.8834839, -167.1984711, 167.2996826

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 131

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2142627, upper bound: 193.2147426
time: 10.17 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2142519, upper bound: 193.2145806
time: 8.30 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -106.8451691, 84.8961182, -88.4172440, 70.3767242, -177.2218933, 173.3133240
1: -90.1857758, 75.6660767, -74.4941483, 62.5548096, -152.7405701, 150.1602020
2: -118.0063858, 76.6207733, -97.6194229, 63.4430885, -181.4494629, 174.2401581
3: -125.2509842, 66.6335373, -103.6093979, 55.1594200, -180.4104004, 170.2429047
4: -114.5754547, 88.1852036, -94.6354141, 72.9312592, -187.5067139, 182.8205719
5: -102.4163361, 80.0481339, -84.7614059, 66.1119919, -168.5283051, 164.8095245
6: -98.4099808, 94.8570175, -81.3930511, 78.4514694, -176.8614349, 176.2500458
7: -107.6129913, 90.0630798, -88.8625946, 74.5482864, -182.1612549, 178.9256439
8: -129.6528168, 88.7157211, -107.3651733, 73.4408569, -203.0936737, 196.0808716
9: -97.5767746, 96.3323364, -80.6630554, 79.6483688, -177.2251434, 176.9953766

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1685207, upper bound: 193.1675240
time: 9.25 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1620140, upper bound: 193.1577501
time: 9.10 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -105.5810852, 83.8747864, -92.6921234, 73.7061081, -179.2871857, 176.5669098
1: -89.1056900, 74.7598801, -78.0234756, 65.5018616, -154.6075439, 152.7833099
2: -116.5994415, 75.7150574, -102.3009262, 66.4473724, -183.0468140, 178.0159912
3: -123.7512131, 65.8374634, -108.6161346, 57.6967430, -181.4479218, 174.4535980
4: -113.2116623, 87.1439590, -99.2308426, 76.4412231, -189.6528778, 186.3748016
5: -101.2036209, 79.0913773, -88.8679352, 69.2391357, -170.4427490, 167.9593201
6: -97.2452698, 93.7209396, -85.3323212, 82.1957703, -179.4410400, 179.0532379
7: -106.3509369, 89.0026855, -93.1742706, 78.1159439, -184.4668427, 182.1769409
8: -128.1008148, 87.6367950, -112.5112534, 76.8174286, -204.9182434, 200.1480255
9: -96.4281235, 95.1855087, -84.5776062, 83.4425049, -179.8706360, 179.7631226

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1652336, upper bound: 193.1656498
time: 8.68 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1603397, upper bound: 193.1571481
time: 7.95 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -86.0575790, 68.5127716, -92.3722687, 73.5199661, -159.5775452, 160.8850403
1: -72.5342712, 60.9147873, -77.7677689, 65.3179855, -137.8522644, 138.6825409
2: -95.0513535, 61.7827606, -102.0459595, 66.2592316, -161.3105774, 163.8287201
3: -100.9227066, 53.7102776, -108.3222198, 57.5232506, -158.4459534, 162.0324860
4: -92.0927048, 71.0233002, -98.8611221, 76.1608429, -168.2535400, 169.8843994
5: -82.4951859, 64.3680420, -88.5591125, 68.9931488, -151.4883270, 152.9271545
6: -79.2368851, 76.3823853, -85.0145340, 81.9109497, -161.1478271, 161.3969116
7: -86.5601578, 72.6045380, -92.8420029, 77.8680038, -164.4281616, 165.4465332
8: -104.5727386, 71.4659042, -112.2190475, 76.5901871, -181.1628723, 183.6849518
9: -78.5413513, 77.5352020, -84.2070847, 83.1270905, -161.6683960, 161.7422791

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2134713, upper bound: 193.2137374
time: 10.44 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2120846, upper bound: 193.2111738
time: 8.25 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -84.5078354, 67.2542114, -96.2079620, 76.5075912, -161.0154266, 163.4621582
1: -71.2027740, 59.7951508, -80.9329224, 67.9591141, -139.1618805, 140.7280426
2: -93.3221741, 60.6664047, -106.2565079, 68.9615860, -162.2837524, 166.9228973
3: -99.0773163, 52.7262611, -112.8200912, 59.7855339, -158.8628540, 165.5463257
4: -90.4169922, 69.7481842, -102.9963303, 79.3236542, -169.7406464, 172.7445068
5: -81.0095139, 63.1883202, -92.2587433, 71.7977829, -152.8072968, 155.4470673
6: -77.8058548, 74.9807205, -88.5606766, 85.2662582, -163.0721130, 163.5413971
7: -85.0089569, 71.3019867, -96.7349625, 81.0808868, -166.0898438, 168.0369415
8: -102.6645203, 70.1348953, -116.8477707, 79.6068802, -182.2713928, 186.9826355
9: -77.1301193, 76.1216965, -87.7360458, 86.5422668, -163.6723785, 163.8577423

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2068931, upper bound: 193.2097707
time: 7.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.2064413, upper bound: 193.2084499
time: 7.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -100.3250351, 79.7401047, -90.7558365, 72.2360992, -172.5611267, 170.4959412
1: -84.6918488, 71.0856552, -76.4229965, 64.1931610, -148.8850098, 147.5086365
2: -110.8300858, 71.9704666, -100.2617264, 65.1036758, -175.9337616, 172.2321930
3: -117.6923370, 62.5919914, -106.4386902, 56.5383186, -174.2306519, 169.0306854
4: -107.5528793, 82.8421326, -97.1318436, 74.8466110, -182.3994751, 179.9739685
5: -96.1317825, 75.1589584, -87.0108719, 67.7966461, -163.9284363, 162.1698303
6: -92.4071426, 89.0568848, -83.5343399, 80.4819565, -172.8890686, 172.5912170
7: -101.0814133, 84.5944366, -91.2229538, 76.5140152, -177.5954285, 175.8173828
8: -121.8124619, 83.2771301, -110.2684097, 75.2724762, -197.0849304, 193.5455017
9: -91.6103668, 90.4445496, -82.7371445, 81.6895447, -173.2998810, 173.1816711

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 246
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1664103, upper bound: 193.1658565
time: 9.90 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1611142, upper bound: 193.1574242
time: 8.21 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -98.8977585, 78.5866852, -94.5407715, 75.1869736, -174.0847321, 173.1274567
1: -83.4724731, 70.0630646, -79.5461044, 66.7999878, -150.2724609, 149.6091614
2: -109.2416306, 70.9469604, -104.4152298, 67.7752304, -177.0168457, 175.3621826
3: -116.0020981, 61.6927071, -110.8789062, 58.7697487, -174.7718506, 172.5715790
4: -106.0149231, 81.6681213, -101.2068100, 77.9667892, -183.9817047, 182.8749237
5: -94.7631149, 74.0780029, -90.6606903, 70.5649872, -165.3280945, 164.7386780
6: -91.0934525, 87.7716293, -87.0318756, 83.7927780, -174.8862000, 174.8034973
7: -99.6574631, 83.3975372, -95.0634460, 79.6883469, -179.3457947, 178.4609680
8: -120.0605164, 82.0580215, -114.8351593, 78.2496338, -198.3101196, 196.8931885
9: -90.3135681, 89.1495438, -86.2226562, 85.0578079, -175.3713684, 175.3721924

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1573686, upper bound: 193.1609426
time: 9.04 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1557482, upper bound: 193.1557482
time: 6.50 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 16.87 seconds
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 2, lower bound: -193.2168512, upper bound: 193.2166849
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 2, lower bound: -193.2165943, upper bound: 193.2161647
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 2, lower bound: -193.2142627, upper bound: 193.2147426
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 2, lower bound: -193.2142519, upper bound: 193.2145806
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 2, lower bound: -193.1685207, upper bound: 193.1675240
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 2, lower bound: -193.1620140, upper bound: 193.1577501
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 2, lower bound: -193.1652336, upper bound: 193.1656498
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 2, lower bound: -193.1603397, upper bound: 193.1571481
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 2, lower bound: -193.2134713, upper bound: 193.2137374
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 2, lower bound: -193.2120846, upper bound: 193.2111738
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 2, lower bound: -193.2068931, upper bound: 193.2097707
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 2, lower bound: -193.2064413, upper bound: 193.2084499
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 2, lower bound: -193.1664103, upper bound: 193.1658565
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 2, lower bound: -193.1611142, upper bound: 193.1574242
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 2, lower bound: -193.1573686, upper bound: 193.1609426
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 2, lower bound: -193.1557482, upper bound: 193.1557482

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -84.2115707, 67.0826492, -84.6466446, 67.4085693, -151.6201477, 151.7292938
1: -70.9174500, 59.5808487, -71.2627487, 59.8751259, -130.7925720, 130.8435974
2: -92.9695892, 60.4353447, -93.4347076, 60.7333755, -153.7029724, 153.8700409
3: -98.7230759, 52.5795250, -99.2094040, 52.8202744, -151.5433350, 151.7889099
4: -90.1416779, 69.4516144, -90.5861053, 69.7943573, -159.9360352, 160.0377197
5: -80.7227478, 62.9762192, -81.1238403, 63.2783890, -144.0011292, 144.1000214
6: -77.5702133, 74.7109909, -77.9409790, 75.0727844, -152.6429749, 152.6519165
7: -84.5542450, 70.9817581, -84.9922104, 71.3395615, -155.8937988, 155.9739685
8: -102.3371277, 70.0271683, -102.8243103, 70.3305893, -172.6677246, 172.8514709
9: -76.7846832, 75.8602905, -77.1722336, 76.2257233, -153.0104065, 153.0325317

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1932605, upper bound: 193.1919524
time: 9.83 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1926083, upper bound: 193.1915989
time: 7.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -86.4189377, 68.8324890, -82.0196533, 65.3157730, -151.7346649, 150.8521423
1: -72.7555466, 61.1314392, -69.0474319, 58.0200043, -130.7755432, 130.1788635
2: -95.3987198, 62.0324173, -90.5188980, 58.8715820, -154.2702942, 152.5513153
3: -101.3703537, 53.9330902, -96.1592178, 51.2017288, -152.5720520, 150.0922699
4: -92.4875793, 71.2608109, -87.7527161, 67.6478729, -160.1354523, 159.0135193
5: -82.8754807, 64.6309509, -78.6025467, 61.3285141, -144.2039795, 143.2334900
6: -79.6426773, 76.6285782, -75.5473862, 72.7305374, -152.3731689, 152.1759644
7: -86.7651672, 72.8629227, -82.3450546, 69.1386185, -155.9037628, 155.2079773
8: -104.9726639, 71.7987518, -99.6530838, 68.1541595, -173.1268311, 171.4518127
9: -78.7784424, 77.8395538, -74.7702637, 73.8704758, -152.6489258, 152.6098175

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1925483, upper bound: 193.1910832
time: 8.99 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1919219, upper bound: 193.1908019
time: 8.70 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -82.8763809, 65.9979172, -89.0869904, 70.8723679, -153.7487335, 155.0848999
1: -69.7708740, 58.6165924, -74.9360809, 62.9442177, -132.7150879, 133.5526733
2: -91.4801178, 59.4750595, -98.3025818, 63.8590927, -155.3392029, 157.7776489
3: -97.1308975, 51.7337036, -104.4143753, 55.4586411, -152.5895233, 156.1480713
4: -88.6976013, 68.3510284, -95.3707428, 73.4434052, -162.1409912, 163.7217712
5: -79.4418335, 61.9604568, -85.3916016, 66.5301132, -145.9719391, 147.3520355
6: -76.3370972, 73.5053101, -82.0316086, 78.9699249, -155.3070068, 155.5368805
7: -83.2182541, 69.8594055, -89.4730606, 75.0502853, -158.2685242, 159.3324585
8: -100.6923981, 68.8802414, -108.1761780, 73.8462906, -174.5386963, 177.0564270
9: -75.5685272, 74.6434555, -81.2382278, 80.1709213, -155.7394409, 155.8816833

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1349089, upper bound: 193.1351616
time: 8.08 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1281974, upper bound: 193.1294209
time: 8.00 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -84.9779739, 67.6612854, -86.4363022, 68.7571106, -153.7350769, 154.0975800
1: -71.5178299, 60.0894966, -72.6991959, 61.0676842, -132.5855103, 132.7886963
2: -93.7893143, 60.9946556, -95.3578415, 61.9787369, -155.7680359, 156.3524780
3: -99.6542587, 53.0213776, -101.3378143, 53.8207474, -153.4749908, 154.3591614
4: -90.9274216, 70.0748444, -92.5046616, 71.2800140, -162.2074280, 162.5794983
5: -81.4947205, 63.5361137, -82.8513412, 64.5597534, -146.0544739, 146.3874512
6: -78.3114471, 75.3259048, -79.6161575, 76.6029282, -154.9143524, 154.9420624
7: -85.3201065, 71.6519775, -86.8012695, 72.8328629, -158.1529694, 158.4532471
8: -103.1966476, 70.5622635, -104.9773712, 71.6410065, -174.8376465, 175.5396423
9: -77.4643936, 76.5261612, -78.8185349, 77.7945404, -155.2589111, 155.3446960

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1331108, upper bound: 193.1332082
time: 8.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1276755, upper bound: 193.1285507
time: 7.20 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -101.6094971, 80.7014313, -85.1665344, 67.7720871, -169.3815765, 165.8679199
1: -85.7517319, 71.9515839, -71.7431564, 60.2485657, -146.0003052, 143.6947327
2: -112.2388229, 72.8696747, -94.0389023, 61.1142082, -173.3530273, 166.9085693
3: -119.1110535, 63.4100876, -99.7969818, 53.1588821, -172.2699127, 163.2070160
4: -108.9155655, 83.8654251, -91.1204376, 70.2496719, -179.1652374, 174.9858551
5: -97.3268356, 76.1148605, -81.6014175, 63.6715508, -160.9983673, 157.7162781
6: -93.5670471, 90.2111435, -78.3852539, 75.5678253, -169.1348724, 168.5964050
7: -102.3789139, 85.6579590, -85.6146088, 71.8132248, -174.1921387, 171.2725525
8: -123.3237000, 84.3477402, -103.4368744, 70.7313538, -194.0550537, 187.7846069
9: -92.7897186, 91.6239624, -77.6916656, 76.7256393, -169.5153351, 169.3156128

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1184856, upper bound: 193.1174620
time: 8.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1162729, upper bound: 193.1153398
time: 8.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -95.6306992, 75.8169937, -74.1030121, 58.9011726, -154.5318756, 149.9199829
1: -80.4830780, 67.5946350, -62.3198128, 52.3743362, -132.8573761, 129.9144440
2: -105.5886154, 68.5077591, -81.8177414, 53.1690674, -158.7576599, 150.3255005
3: -112.1145172, 59.6584091, -86.8312302, 46.3505592, -158.4650726, 146.4895782
4: -102.3740158, 78.8197937, -79.1724854, 61.1036415, -163.4776611, 157.9922791
5: -91.3675232, 71.4684525, -70.8493500, 55.3434677, -146.7109680, 142.3177948
6: -87.9795837, 84.8205719, -68.1601105, 65.7145538, -153.6941376, 152.9806671
7: -96.3319168, 80.5442505, -74.5046234, 62.4761543, -158.8080597, 155.0488739
8: -116.1064606, 79.1935349, -90.0440826, 61.4666176, -177.5730591, 169.2376099
9: -87.2524033, 86.1798477, -67.5318069, 66.7595901, -154.0119781, 153.7116547

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0988093, upper bound: 193.0922039
time: 8.02 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0980134, upper bound: 193.0918716
time: 8.12 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -100.3402939, 79.6757278, -89.4645081, 71.1202469, -171.4605255, 169.1402283
1: -84.6665649, 71.0412827, -75.2911377, 63.2114220, -147.8779602, 146.3324280
2: -110.8257828, 71.9605026, -98.7458572, 64.1346588, -174.9604492, 170.7063293
3: -117.6054993, 62.6107826, -104.8307724, 55.7091408, -173.3146362, 167.4415436
4: -107.5466309, 82.8208008, -95.7419662, 73.7791595, -181.3257904, 178.5627747
5: -96.1089096, 75.1547546, -85.7305603, 66.8160629, -162.9249725, 160.8852997
6: -92.3977814, 89.0703125, -82.3461838, 79.3325348, -171.7303162, 171.4164886
7: -101.1121902, 84.5936737, -89.9505615, 75.4002304, -176.5124054, 174.5442352
8: -121.7662735, 83.2640076, -108.6111145, 74.1263809, -195.8926544, 191.8751221
9: -91.6367188, 90.4727402, -81.6280060, 80.5405502, -172.1772766, 172.1006927

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1143232, upper bound: 193.1143263
time: 8.32 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.1100301, upper bound: 193.1109257
time: 9.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -94.4089127, 74.8299713, -78.4585724, 62.2969856, -156.7058868, 153.2885284
1: -79.4404144, 66.7183990, -65.9165039, 55.3792686, -134.8196564, 132.6348724
2: -104.2281342, 67.6295853, -86.5901947, 56.2316666, -160.4597626, 154.2197876
3: -110.6651306, 58.8881073, -91.9331055, 48.9332924, -159.5984192, 150.8211975
4: -101.0555344, 77.8130112, -83.8548431, 64.6834717, -165.7389832, 161.6678467
5: -90.1964493, 70.5432816, -75.0342865, 58.5341835, -148.7306366, 145.5775757
6: -86.8534393, 83.7240601, -72.1748505, 69.5324402, -156.3858643, 155.8989105
7: -95.1108627, 79.5179749, -78.9007034, 66.1141891, -161.2250519, 158.4186707
8: -114.6079712, 78.1516647, -95.2926331, 64.9132538, -179.5212250, 173.4443054
9: -86.1401291, 85.0707626, -71.5239487, 70.6295090, -156.7696381, 156.5946960

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0968358, upper bound: 193.0914844
time: 7.27 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -193.0950397, upper bound: 193.0907842
time: 7.56 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -78.7802124, 62.7235870, -87.0753098, 69.3069687, -148.0871735, 149.7988892
1: -66.4051590, 55.7475700, -73.3069458, 61.5567856, -127.9619446, 129.0545197
2: -87.0089111, 56.5842781, -96.1913910, 62.4733582, -149.4822540, 152.7756500
3: -92.3848419, 49.2587433, -102.1073608, 54.2823410, -146.6671753, 151.3660736
4: -84.2221680, 65.0268860, -93.1346207, 71.7960205, -156.0181885, 158.1614990
5: -75.4724579, 58.9051285, -83.4478683, 65.0165482, -140.4889984, 142.3529968
6: -72.4678421, 69.9303131, -80.0878754, 77.2143707, -149.6821747, 150.0181580
7: -79.2590637, 66.5131073, -87.5274582, 73.4328308, -152.6918945, 154.0405273
8: -95.7711334, 65.4034729, -105.8110504, 72.1800842, -167.9512177, 171.2145233
9: -71.8869629, 70.9540863, -79.3632812, 78.3369217, -150.2238770, 150.3173676

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 131

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1935479, upper bound: 193.1937137
time: 9.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1935227, upper bound: 193.1936391
time: 9.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -79.6155853, 63.3266296, -83.2767563, 66.2650223, -145.8805847, 146.6033936
1: -67.0482254, 56.2838783, -70.0907288, 58.8578873, -125.9061050, 126.3746033
2: -87.9087372, 57.1355324, -91.9779739, 59.7510605, -147.6597748, 149.1135101
3: -93.3870621, 49.8005295, -97.6501389, 51.9628372, -145.3498993, 147.4506683
4: -85.0982590, 65.6418762, -89.0206833, 68.6545334, -153.7527924, 154.6625671
5: -76.2200394, 59.4624825, -79.7725830, 62.1696053, -138.3896484, 139.2350616
6: -73.1913528, 70.6484222, -76.5605621, 73.8419418, -147.0332947, 147.2089691
7: -80.1034927, 67.2104797, -83.7166595, 70.2515182, -150.3550110, 150.9271393
8: -96.7241364, 65.9685974, -101.1961060, 68.9970169, -165.7211304, 167.1647034
9: -72.6317139, 71.6308899, -75.8986206, 74.8942108, -147.5259247, 147.5294647

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 246

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 131

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1914585, upper bound: 193.1905741
time: 9.22 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -193.1914885, upper bound: 193.1905243
time: 8.38 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 18.94 seconds
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.1932605, upper bound: 193.1919524
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.1926083, upper bound: 193.1915989
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.1925483, upper bound: 193.1910832
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.1919219, upper bound: 193.1908019
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.1349089, upper bound: 193.1351616
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.1281974, upper bound: 193.1294209
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.1331108, upper bound: 193.1332082
IS_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.1276755, upper bound: 193.1285507
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.1184856, upper bound: 193.1174620
IS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.1162729, upper bound: 193.1153398
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.0988093, upper bound: 193.0922039
IS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.0980134, upper bound: 193.0918716
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.1143232, upper bound: 193.1143263
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.1100301, upper bound: 193.1109257
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.0968358, upper bound: 193.0914844
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.0950397, upper bound: 193.0907842
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.1935479, upper bound: 193.1937137
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.1935227, upper bound: 193.1936391
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.1914585, upper bound: 193.1905741
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.94
Output dim: 2, lower bound: -193.1914885, upper bound: 193.1905243
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 2, lower bound: -193.2068931, upper bound: 193.2097707
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 2, lower bound: -193.2064413, upper bound: 193.2084499
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 2, lower bound: -193.1664103, upper bound: 193.1658565
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 2, lower bound: -193.1611142, upper bound: 193.1574242
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 2, lower bound: -193.1573686, upper bound: 193.1609426
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 2, lower bound: -193.1557482, upper bound: 193.1557482
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=194.859130859375
rel_dist={2: [-193.2885465354435, 193.28854653544352]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1817.88 seconds
