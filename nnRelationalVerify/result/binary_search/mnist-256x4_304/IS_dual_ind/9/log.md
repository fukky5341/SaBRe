## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 197.2433907684
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486)
1: (-87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961)
2: (-114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986)
3: (-122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255)
4: (-112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691)
5: (-100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425)
6: (-96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658)
7: (-105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948)
8: (-125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392)
9: (-96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598)

## BASE Result
execution time: IAR + LP analysis = 1.45 + 8.18 = 9.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -197.4409435, upper bound: 197.4409435


# Binary Search by BASE starts (time budget: 1990.36 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=198.953369140625
rel_dist={4: [-197.44087218970873, 197.4408721892934]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=198.953369140625
rel_dist={4: [-197.44083159618555, 197.44083163160866]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=198.953369140625
rel_dist={4: [-197.4407374020123, 197.4407374020123]}

## Binary Search Result
Binary search time: 33.06 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1957.30 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4139380, upper bound: 197.4149538
time: 7.90 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4388910, upper bound: 197.4388910
time: 5.40 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.46 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.46
Output dim: 4, lower bound: -197.4139380, upper bound: 197.4149538
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.46
Output dim: 4, lower bound: -197.4388910, upper bound: 197.4388910

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -79.9444504, 63.3750114, -100.2639008, 79.5944519, -159.5389099, 163.6389160
1: -66.2200775, 56.2719460, -83.5150375, 70.6974564, -136.9174957, 139.7869873
2: -87.7009277, 57.5595970, -110.2355957, 72.0445023, -159.7454224, 167.7951965
3: -93.7524338, 49.1665268, -117.4377365, 61.6957703, -155.4482117, 166.6042480
4: -85.9368134, 66.0450745, -107.8952103, 82.9497681, -168.8865814, 173.9402771
5: -76.8184204, 59.8392181, -96.2967300, 75.0339584, -151.8523712, 156.1359558
6: -74.1077042, 70.4479294, -92.7775345, 88.5808029, -162.6885071, 163.2254486
7: -80.7441788, 67.8362045, -101.1543808, 84.9008255, -165.6449738, 168.9905701
8: -96.0069962, 65.3933868, -120.8039017, 82.5889969, -178.5959930, 186.1972504
9: -73.5597076, 71.9806671, -92.2732697, 90.5586166, -164.1183167, 164.2539368

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4035856, upper bound: 197.4035856
time: 5.63 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4035856, upper bound: 197.4149538
time: 5.88 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -99.1355057, 78.7032318, -104.5059052, 82.9751511, -182.1105957, 183.2091217
1: -82.5673141, 69.9038849, -87.1186676, 73.7036362, -156.2709503, 157.0225525
2: -108.9954758, 71.2377625, -114.9358521, 75.0652771, -184.0607605, 186.1735687
3: -116.1185074, 61.0004578, -122.3928833, 64.3120346, -180.4305267, 183.3933105
4: -106.6865845, 82.0365067, -112.4815369, 86.4718399, -193.1584015, 194.5180206
5: -95.2138977, 74.1955032, -100.3661041, 78.2024307, -173.4163208, 174.5616150
6: -91.7468567, 87.5859756, -96.6764297, 92.3622818, -184.1091309, 184.2623901
7: -100.0214844, 83.9518967, -105.4159775, 88.4643326, -188.4857635, 189.3678741
8: -119.4582748, 81.6738663, -125.9699860, 86.1614532, -205.6197205, 207.6438599
9: -91.2429123, 89.5477371, -96.1728745, 94.4252930, -185.6681976, 185.7206116

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4149538, upper bound: 197.4139380
time: 6.86 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4149538, upper bound: 197.4388910
time: 6.86 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 15.20 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 15.20
Output dim: 4, lower bound: -197.4035856, upper bound: 197.4035856
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 15.20
Output dim: 4, lower bound: -197.4035856, upper bound: 197.4149538
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 15.20
Output dim: 4, lower bound: -197.4149538, upper bound: 197.4139380
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 15.20
Output dim: 4, lower bound: -197.4149538, upper bound: 197.4388910

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -79.9444504, 63.3750114, -79.9444504, 63.3750114, -143.3194580, 143.3194580
1: -66.2200775, 56.2719460, -66.2200775, 56.2719460, -122.4920120, 122.4920120
2: -87.7009277, 57.5595970, -87.7009277, 57.5595970, -145.2605133, 145.2605133
3: -93.7524338, 49.1665268, -93.7524338, 49.1665268, -142.9189606, 142.9189606
4: -85.9368134, 66.0450745, -85.9368134, 66.0450745, -151.9818878, 151.9818878
5: -76.8184204, 59.8392181, -76.8184204, 59.8392181, -136.6576385, 136.6576385
6: -74.1077042, 70.4479294, -74.1077042, 70.4479294, -144.5556335, 144.5556335
7: -80.7441788, 67.8362045, -80.7441788, 67.8362045, -148.5803528, 148.5803528
8: -96.0069962, 65.3933868, -96.0069962, 65.3933868, -161.4003906, 161.4003906
9: -73.5597076, 71.9806671, -73.5597076, 71.9806671, -145.5403595, 145.5403595

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3978577, upper bound: 197.3977747
time: 4.40 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3979463, upper bound: 197.3979463
time: 6.06 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -79.9444504, 63.3750114, -99.1355057, 78.7032318, -158.6476593, 162.5104828
1: -66.2200775, 56.2719460, -82.5673141, 69.9038849, -136.1239624, 138.8392639
2: -87.7009277, 57.5595970, -108.9954758, 71.2377625, -158.9386597, 166.5550690
3: -93.7524338, 49.1665268, -116.1185074, 61.0004578, -154.7528839, 165.2850037
4: -85.9368134, 66.0450745, -106.6865845, 82.0365067, -167.9733124, 172.7316589
5: -76.8184204, 59.8392181, -95.2138977, 74.1955032, -151.0139160, 155.0531158
6: -74.1077042, 70.4479294, -91.7468567, 87.5859756, -161.6936798, 162.1947937
7: -80.7441788, 67.8362045, -100.0214844, 83.9518967, -164.6960754, 167.8576660
8: -96.0069962, 65.3933868, -119.4582748, 81.6738663, -177.6808624, 184.8516235
9: -73.5597076, 71.9806671, -91.2429123, 89.5477371, -163.1074371, 163.2235718

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3978577, upper bound: 197.4093416
time: 4.57 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3979463, upper bound: 197.4095034
time: 5.76 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -99.1355057, 78.7032318, -79.9444504, 63.3750114, -162.5104828, 158.6476593
1: -82.5673141, 69.9038849, -66.2200775, 56.2719460, -138.8392639, 136.1239624
2: -108.9954758, 71.2377625, -87.7009277, 57.5595970, -166.5550690, 158.9386597
3: -116.1185074, 61.0004578, -93.7524338, 49.1665268, -165.2850037, 154.7528839
4: -106.6865845, 82.0365067, -85.9368134, 66.0450745, -172.7316589, 167.9733124
5: -95.2138977, 74.1955032, -76.8184204, 59.8392181, -155.0531158, 151.0139160
6: -91.7468567, 87.5859756, -74.1077042, 70.4479294, -162.1947937, 161.6936798
7: -100.0214844, 83.9518967, -80.7441788, 67.8362045, -167.8576660, 164.6960754
8: -119.4582748, 81.6738663, -96.0069962, 65.3933868, -184.8516235, 177.6808624
9: -91.2429123, 89.5477371, -73.5597076, 71.9806671, -163.2235718, 163.1074371

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4092035, upper bound: 197.4082723
time: 7.11 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4095035, upper bound: 197.4086520
time: 6.41 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -99.1355057, 78.7032318, -99.1355057, 78.7032318, -177.8386688, 177.8386688
1: -82.5673141, 69.9038849, -82.5673141, 69.9038849, -152.4711609, 152.4711609
2: -108.9954758, 71.2377625, -108.9954758, 71.2377625, -180.2332306, 180.2332306
3: -116.1185074, 61.0004578, -116.1185074, 61.0004578, -177.1189117, 177.1189117
4: -106.6865845, 82.0365067, -106.6865845, 82.0365067, -188.7230530, 188.7230530
5: -95.2138977, 74.1955032, -95.2138977, 74.1955032, -169.4093933, 169.4093933
6: -91.7468567, 87.5859756, -91.7468567, 87.5859756, -179.3328247, 179.3328247
7: -100.0214844, 83.9518967, -100.0214844, 83.9518967, -183.9733887, 183.9733887
8: -119.4582748, 81.6738663, -119.4582748, 81.6738663, -201.1321259, 201.1321259
9: -91.2429123, 89.5477371, -91.2429123, 89.5477371, -180.7906494, 180.7906494

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4092035, upper bound: 197.4349830
time: 6.92 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4095035, upper bound: 197.4353516
time: 6.93 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 15.40 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 4, lower bound: -197.3978577, upper bound: 197.3977747
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 4, lower bound: -197.3979463, upper bound: 197.3979463
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 4, lower bound: -197.3978577, upper bound: 197.4093416
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 4, lower bound: -197.3979463, upper bound: 197.4095034
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 4, lower bound: -197.4092035, upper bound: 197.4082723
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 4, lower bound: -197.4095035, upper bound: 197.4086520
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 4, lower bound: -197.4092035, upper bound: 197.4349830
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 4, lower bound: -197.4095035, upper bound: 197.4353516

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -62.2587967, 49.4172249, -78.4439545, 62.1901321, -124.4489212, 127.8611755
1: -51.1962242, 43.7674103, -64.9446030, 55.2099800, -106.4062042, 108.7119904
2: -68.1837463, 44.9291420, -86.0434113, 56.4875145, -124.6712570, 130.9725342
3: -73.0340500, 38.2486572, -91.9934921, 48.2393036, -121.2733459, 130.2421265
4: -66.9400177, 51.4695473, -84.3246384, 64.8064728, -131.7464600, 135.7941895
5: -59.9183273, 46.6845436, -75.3845749, 58.7223663, -118.6406937, 122.0691147
6: -57.8632851, 54.7351456, -72.7288208, 69.1138458, -126.9771118, 127.4639664
7: -62.9254227, 52.9625816, -79.2303391, 66.5737686, -129.4991913, 132.1928864
8: -74.6212311, 50.7602539, -94.1915817, 64.1501389, -138.7713470, 144.9518433
9: -57.2352982, 55.8247681, -72.1727448, 70.6078339, -127.8431168, 127.9974976

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3942882, upper bound: 197.3936674
time: 5.81 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3956725, upper bound: 197.3955896
time: 5.17 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -68.3321838, 54.2141838, -79.3037109, 62.8693542, -131.2015228, 133.5178986
1: -56.3511658, 48.0633049, -65.6756363, 55.8189659, -112.1701126, 113.7389374
2: -74.8788223, 49.2853546, -86.9932175, 57.1029968, -131.9817963, 136.2785645
3: -80.1520615, 41.9963646, -93.0017548, 48.7709198, -128.9229584, 134.9981079
4: -73.4675751, 56.4845314, -85.2488556, 65.5172577, -138.9848328, 141.7333679
5: -65.7274704, 51.2121849, -76.2064056, 59.3631248, -125.0905914, 127.4185944
6: -63.4669647, 60.1261253, -73.5204315, 69.8783493, -133.3453064, 133.6465607
7: -69.0598907, 58.1050606, -80.0991440, 67.2990875, -136.3589630, 138.2042084
8: -81.9763718, 55.7643127, -95.2328033, 64.8619614, -146.8383331, 150.9970856
9: -62.8733368, 61.4070702, -72.9696198, 71.3970795, -134.2704163, 134.3766937

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3944233, upper bound: 197.3939268
time: 5.79 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3957764, upper bound: 197.3957764
time: 4.57 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -62.2587967, 49.4172249, -97.5970993, 77.4883652, -139.7471619, 147.0143280
1: -51.1962242, 43.7674103, -81.2596741, 68.8159103, -120.0121307, 125.0270691
2: -68.1837463, 44.9291420, -107.2967834, 70.1393127, -138.3230591, 152.2259064
3: -73.0340500, 38.2486572, -114.3142014, 60.0496368, -133.0836792, 152.5628204
4: -66.9400177, 51.4695473, -105.0335999, 80.7668839, -147.7069092, 156.5031128
5: -59.9183273, 46.6845436, -93.7431641, 73.0505829, -132.9688568, 140.4276886
6: -57.8632851, 54.7351456, -90.3329468, 86.2187271, -144.0820160, 145.0680847
7: -62.9254227, 52.9625816, -98.4695053, 82.6574020, -145.5827942, 151.4320526
8: -74.6212311, 50.7602539, -117.5987778, 80.3998795, -155.0211029, 168.3590393
9: -57.2352982, 55.8247681, -89.8216095, 88.1416626, -145.3769531, 145.6463776

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4082437, upper bound: 197.4091650
time: 7.31 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4082437, upper bound: 197.4093383
time: 7.31 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -68.3321838, 54.2141838, -98.5026093, 78.2038345, -146.5359955, 152.7167511
1: -56.3511658, 48.0633049, -82.0298691, 69.4567871, -125.8079453, 130.0931702
2: -74.8788223, 49.2853546, -108.2966309, 70.7869873, -145.6658020, 157.5819855
3: -80.1520615, 41.9963646, -115.3766861, 60.6095428, -140.7615967, 157.3730469
4: -73.4675751, 56.4845314, -106.0070877, 81.5151749, -154.9827576, 162.4916229
5: -65.7274704, 51.2121849, -94.6091614, 73.7252121, -139.4526825, 145.8213348
6: -63.4669647, 60.1261253, -91.1664200, 87.0234146, -150.4903870, 151.2925415
7: -69.0598907, 58.1050606, -99.3839722, 83.4210358, -152.4809265, 157.4890289
8: -81.9763718, 55.7643127, -118.6940994, 81.1498718, -163.1262512, 174.4584045
9: -62.8733368, 61.4070702, -90.6603012, 88.9714127, -151.8447113, 152.0673676

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4082722, upper bound: 197.4092035
time: 6.61 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4082722, upper bound: 197.4095034
time: 6.35 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -80.9900284, 64.3731308, -78.4439545, 62.1901321, -143.1801147, 142.8170776
1: -67.1509247, 57.0728683, -64.9446030, 55.2099800, -122.3609009, 122.0174332
2: -88.9628677, 58.2864723, -86.0434113, 56.4875145, -145.4503784, 144.3298645
3: -94.8495255, 49.7938385, -91.9934921, 48.2393036, -143.0888062, 141.7873230
4: -87.1935120, 67.0708923, -84.3246384, 64.8064728, -151.9999695, 151.3955383
5: -77.8680878, 60.6965294, -75.3845749, 58.7223663, -136.5904541, 136.0810852
6: -75.0712967, 71.4620056, -72.7288208, 69.1138458, -144.1851501, 144.1908264
7: -81.7263260, 68.6844559, -79.2303391, 66.5737686, -148.3000946, 147.9147644
8: -97.5274811, 66.6552429, -94.1915817, 64.1501389, -161.6776123, 160.8468170
9: -74.4901352, 72.9769440, -72.1727448, 70.6078339, -145.0979614, 145.1496887

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4054953, upper bound: 197.4042846
time: 7.35 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4067875, upper bound: 197.4061057
time: 5.56 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -87.6675110, 69.6542511, -79.3037109, 62.8693542, -150.5368652, 148.9579620
1: -72.8256454, 61.8019753, -65.6756363, 55.8189659, -128.6446075, 127.4776077
2: -96.3304672, 63.0746346, -86.9932175, 57.1029968, -153.4334717, 150.0678558
3: -102.6804047, 53.9181709, -93.0017548, 48.7709198, -151.4512787, 146.9198914
4: -94.3752365, 72.5914841, -85.2488556, 65.5172577, -159.8924866, 157.8403168
5: -84.2571259, 65.6748962, -76.2064056, 59.3631248, -143.6202545, 141.8813019
6: -81.2340698, 77.3911514, -73.5204315, 69.8783493, -151.1123962, 150.9115906
7: -88.4754868, 74.3349457, -80.0991440, 67.2990875, -155.7745514, 154.4340820
8: -105.6106262, 72.1767273, -95.2328033, 64.8619614, -170.4725800, 167.4095154
9: -80.6891785, 79.1061096, -72.9696198, 71.3970795, -152.0862579, 152.0757141

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4058066, upper bound: 197.4047014
time: 6.94 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4071038, upper bound: 197.4065540
time: 7.42 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -80.9900284, 64.3731308, -97.5970993, 77.4883652, -158.4783936, 161.9702301
1: -67.1509247, 57.0728683, -81.2596741, 68.8159103, -135.9668274, 138.3325500
2: -88.9628677, 58.2864723, -107.2967834, 70.1393127, -159.1021729, 165.5832367
3: -94.8495255, 49.7938385, -114.3142014, 60.0496368, -154.8991699, 164.1080322
4: -87.1935120, 67.0708923, -105.0335999, 80.7668839, -167.9603882, 172.1044922
5: -77.8680878, 60.6965294, -93.7431641, 73.0505829, -150.9186401, 154.4396973
6: -75.0712967, 71.4620056, -90.3329468, 86.2187271, -161.2900238, 161.7949524
7: -81.7263260, 68.6844559, -98.4695053, 82.6574020, -164.3837128, 167.1539459
8: -97.5274811, 66.6552429, -117.5987778, 80.3998795, -177.9273529, 184.2540283
9: -74.4901352, 72.9769440, -89.8216095, 88.1416626, -162.6318054, 162.7985535

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4348344, upper bound: 197.4348510
time: 5.81 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4348344, upper bound: 197.4349829
time: 5.24 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -87.6675110, 69.6542511, -98.5026093, 78.2038345, -165.8713379, 168.1568604
1: -72.8256454, 61.8019753, -82.0298691, 69.4567871, -142.2824402, 143.8318329
2: -96.3304672, 63.0746346, -108.2966309, 70.7869873, -167.1174622, 171.3712616
3: -102.6804047, 53.9181709, -115.3766861, 60.6095428, -163.2899323, 169.2948303
4: -94.3752365, 72.5914841, -106.0070877, 81.5151749, -175.8904114, 178.5985565
5: -84.2571259, 65.6748962, -94.6091614, 73.7252121, -157.9823303, 160.2840424
6: -81.2340698, 77.3911514, -91.1664200, 87.0234146, -168.2574768, 168.5575714
7: -88.4754868, 74.3349457, -99.3839722, 83.4210358, -171.8965149, 173.7189178
8: -105.6106262, 72.1767273, -118.6940994, 81.1498718, -186.7604980, 190.8708191
9: -80.6891785, 79.1061096, -90.6603012, 88.9714127, -169.6605682, 169.7664032

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4349782, upper bound: 197.4350184
time: 5.51 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4349782, upper bound: 197.4353516
time: 5.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 12.88 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.88
Output dim: 4, lower bound: -197.3942882, upper bound: 197.3936674
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.88
Output dim: 4, lower bound: -197.3956725, upper bound: 197.3955896
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.88
Output dim: 4, lower bound: -197.3944233, upper bound: 197.3939268
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.88
Output dim: 4, lower bound: -197.3957764, upper bound: 197.3957764
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.88
Output dim: 4, lower bound: -197.4082437, upper bound: 197.4091650
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.88
Output dim: 4, lower bound: -197.4082437, upper bound: 197.4093383
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.88
Output dim: 4, lower bound: -197.4082722, upper bound: 197.4092035
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.88
Output dim: 4, lower bound: -197.4082722, upper bound: 197.4095034
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.88
Output dim: 4, lower bound: -197.4054953, upper bound: 197.4042846
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.88
Output dim: 4, lower bound: -197.4067875, upper bound: 197.4061057
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.88
Output dim: 4, lower bound: -197.4058066, upper bound: 197.4047014
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.88
Output dim: 4, lower bound: -197.4071038, upper bound: 197.4065540
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.88
Output dim: 4, lower bound: -197.4348344, upper bound: 197.4348510
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.88
Output dim: 4, lower bound: -197.4348344, upper bound: 197.4349829
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.88
Output dim: 4, lower bound: -197.4349782, upper bound: 197.4350184
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.88
Output dim: 4, lower bound: -197.4349782, upper bound: 197.4353516

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -61.2904625, 48.6561813, -63.3569565, 50.3229637, -111.6134109, 112.0131226
1: -50.3931847, 43.0922356, -52.4208069, 44.6792297, -95.0724182, 95.5130386
2: -67.1210556, 44.2446136, -69.4866028, 45.8237076, -112.9447632, 113.7312088
3: -71.9039688, 37.6542931, -74.3834381, 38.9929352, -110.8969040, 112.0377197
4: -65.9038773, 50.6826591, -68.1608353, 52.5523834, -118.4562607, 118.8434906
5: -58.9940643, 45.9702759, -60.9801407, 47.5998383, -106.5938950, 106.9504166
6: -56.9826431, 53.8823471, -58.9809036, 55.8263893, -112.8090210, 112.8632431
7: -61.9566383, 52.1603470, -64.1327133, 54.0716095, -116.0282440, 116.2930603
8: -73.4594879, 49.9643021, -76.0788040, 51.7647820, -125.2242737, 126.0431061
9: -56.3528900, 54.9494057, -58.4306831, 57.0006294, -113.3535156, 113.3800888

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3930257, upper bound: 197.3929668
time: 5.91 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3930257, upper bound: 197.3936674
time: 6.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -61.9798813, 49.1979637, -71.2988281, 56.5714684, -118.5513382, 120.4967957
1: -50.9644241, 43.5729485, -58.9950790, 50.2210693, -101.1854935, 102.5680237
2: -67.8778229, 44.7321053, -78.2026062, 51.4425201, -119.3203430, 122.9347076
3: -72.7087860, 38.0777779, -83.6586075, 43.8634796, -116.5722580, 121.7363815
4: -66.6415787, 51.2425003, -76.6757889, 58.9907455, -125.6323242, 127.9182739
5: -59.6517830, 46.4788742, -68.5557098, 53.4519424, -113.1037292, 115.0345764
6: -57.6093369, 54.4894600, -66.2247238, 62.8158684, -120.4251938, 120.7141876
7: -62.6469231, 52.7312164, -72.0952911, 60.6475906, -123.2945099, 124.8265076
8: -74.2866669, 50.5311089, -85.6167450, 58.2783546, -132.5650177, 136.1478424
9: -56.9809799, 55.5730019, -65.6587906, 64.1624985, -121.1434784, 121.2317963

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3937216, upper bound: 197.3941758
time: 6.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3937216, upper bound: 197.3955896
time: 5.48 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -67.3441620, 53.4371872, -64.2197647, 51.0040817, -118.3482437, 117.6569519
1: -55.5301476, 47.3731766, -53.1545982, 45.2902412, -100.8203812, 100.5277710
2: -73.7940750, 48.5872002, -70.4391861, 46.4416122, -120.2356796, 119.0263824
3: -78.9986115, 41.3899879, -75.3948669, 39.5263748, -118.5249786, 116.7848511
4: -72.4090424, 55.6819038, -69.0884781, 53.2651253, -125.6741638, 124.7703705
5: -64.7836075, 50.4831581, -61.8047142, 48.2426605, -113.0262680, 112.2878723
6: -62.5682297, 59.2552109, -59.7748184, 56.5934372, -119.1616592, 119.0300217
7: -68.0716171, 57.2860718, -65.0039978, 54.7991371, -122.8707581, 122.2900696
8: -80.7906418, 54.9526939, -77.1237946, 52.4790840, -133.2697296, 132.0764618
9: -61.9729881, 60.5144119, -59.2300949, 57.7926559, -119.7656403, 119.7444916

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3932323, upper bound: 197.3932323
time: 5.20 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3932323, upper bound: 197.3939268
time: 5.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -68.0464172, 53.9893150, -72.1585693, 57.2499733, -125.2963867, 126.1478882
1: -56.1131592, 47.8637161, -59.7260551, 50.8299408, -106.9431000, 107.5897675
2: -74.5650711, 49.0835686, -79.1513901, 52.0581474, -126.6232147, 128.2349243
3: -79.8186493, 41.8214073, -84.6666565, 44.3948631, -124.2135086, 126.4880600
4: -73.1613007, 56.2519569, -77.5997849, 59.7008934, -132.8621979, 133.8517456
5: -65.4542313, 51.0013008, -69.3774261, 54.0924072, -119.5466385, 120.3787231
6: -63.2067680, 59.8740082, -67.0160065, 63.5798798, -126.7866364, 126.8900146
7: -68.7744675, 57.8679428, -72.9633865, 61.3727074, -130.1471710, 130.8313141
8: -81.6329498, 55.5295448, -86.6569061, 58.9899521, -140.6228943, 142.1864471
9: -62.6128082, 61.1490288, -66.4556046, 64.9511642, -127.5639725, 127.6046295

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3939268, upper bound: 197.3944233
time: 5.99 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3939268, upper bound: 197.3957764
time: 6.16 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -62.2587967, 49.4172249, -80.9900284, 64.3731308, -126.6319275, 130.4072418
1: -51.1962242, 43.7674103, -67.1509247, 57.0728683, -108.2690811, 110.9183197
2: -68.1837463, 44.9291420, -88.9628677, 58.2864723, -126.4701996, 133.8919983
3: -73.0340500, 38.2486572, -94.8495255, 49.7938385, -122.8278809, 133.0981598
4: -66.9400177, 51.4695473, -87.1935120, 67.0708923, -134.0109100, 138.6630402
5: -59.9183273, 46.6845436, -77.8680878, 60.6965294, -120.6148529, 124.5526276
6: -57.8632851, 54.7351456, -75.0712967, 71.4620056, -129.3252869, 129.8064423
7: -62.9254227, 52.9625816, -81.7263260, 68.6844559, -131.6098633, 134.6889038
8: -74.6212311, 50.7602539, -97.5274811, 66.6552429, -141.2764587, 148.2877350
9: -57.2352982, 55.8247681, -74.4901352, 72.9769440, -130.2122498, 130.3148956

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4043383, upper bound: 197.4054724
time: 6.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4061244, upper bound: 197.4067806
time: 7.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -62.2587967, 49.4172249, -87.6675110, 69.6542511, -131.9130554, 137.0847321
1: -51.1962242, 43.7674103, -72.8256454, 61.8019753, -112.9981995, 116.5930405
2: -68.1837463, 44.9291420, -96.3304672, 63.0746346, -131.2583771, 141.2596130
3: -73.0340500, 38.2486572, -102.6804047, 53.9181709, -126.9522247, 140.9290161
4: -66.9400177, 51.4695473, -94.3752365, 72.5914841, -139.5314789, 145.8447723
5: -59.9183273, 46.6845436, -84.2571259, 65.6748962, -125.5932236, 130.9416351
6: -57.8632851, 54.7351456, -81.2340698, 77.3911514, -135.2544098, 135.9692078
7: -62.9254227, 52.9625816, -88.4754868, 74.3349457, -137.2603760, 141.4380493
8: -74.6212311, 50.7602539, -105.6106262, 72.1767273, -146.7979431, 156.3708801
9: -57.2352982, 55.8247681, -80.6891785, 79.1061096, -136.3413849, 136.5139313

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4043383, upper bound: 197.4056164
time: 8.19 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4061244, upper bound: 197.4069456
time: 5.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -68.3321838, 54.2141838, -80.9900284, 64.3731308, -132.7052917, 135.2041779
1: -56.3511658, 48.0633049, -67.1509247, 57.0728683, -113.4240036, 115.2142334
2: -74.8788223, 49.2853546, -88.9628677, 58.2864723, -133.1652527, 138.2482300
3: -80.1520615, 41.9963646, -94.8495255, 49.7938385, -129.9458923, 136.8458862
4: -73.4675751, 56.4845314, -87.1935120, 67.0708923, -140.5384674, 143.6780396
5: -65.7274704, 51.2121849, -77.8680878, 60.6965294, -126.4239960, 129.0802765
6: -63.4669647, 60.1261253, -75.0712967, 71.4620056, -134.9289703, 135.1974182
7: -69.0598907, 58.1050606, -81.7263260, 68.6844559, -137.7443542, 139.8313904
8: -81.9763718, 55.7643127, -97.5274811, 66.6552429, -148.6316223, 153.2917633
9: -62.8733368, 61.4070702, -74.4901352, 72.9769440, -135.8502808, 135.8972015

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4042846, upper bound: 197.4054953
time: 6.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4061057, upper bound: 197.4067875
time: 7.44 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -68.3321838, 54.2141838, -87.6675110, 69.6542511, -137.9864349, 141.8816833
1: -56.3511658, 48.0633049, -72.8256454, 61.8019753, -118.1531296, 120.8889465
2: -74.8788223, 49.2853546, -96.3304672, 63.0746346, -137.9534302, 145.6158142
3: -80.1520615, 41.9963646, -102.6804047, 53.9181709, -134.0702057, 144.6767578
4: -73.4675751, 56.4845314, -94.3752365, 72.5914841, -146.0590515, 150.8597717
5: -65.7274704, 51.2121849, -84.2571259, 65.6748962, -131.4023743, 135.4693146
6: -63.4669647, 60.1261253, -81.2340698, 77.3911514, -140.8581238, 141.3601837
7: -69.0598907, 58.1050606, -88.4754868, 74.3349457, -143.3948364, 146.5805359
8: -81.9763718, 55.7643127, -105.6106262, 72.1767273, -154.1531067, 161.3749084
9: -62.8733368, 61.4070702, -80.6891785, 79.1061096, -141.9794312, 142.0962524

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4042846, upper bound: 197.4057881
time: 6.40 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4061057, upper bound: 197.4070641
time: 7.89 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -80.0319366, 63.6196327, -63.3569565, 50.3229637, -130.3549042, 126.9765854
1: -66.3554535, 56.4039536, -52.4208069, 44.6792297, -111.0346832, 108.8247528
2: -87.9108963, 57.6097527, -69.4866028, 45.8237076, -133.7346039, 127.0963516
3: -93.7304688, 49.2056313, -74.3834381, 38.9929352, -132.7233887, 123.5890579
4: -86.1683121, 66.2925720, -68.1608353, 52.5523834, -138.7206879, 134.4533997
5: -76.9532928, 59.9896011, -60.9801407, 47.5998383, -124.5531158, 120.9697266
6: -74.1995010, 70.6179428, -58.9809036, 55.8263893, -130.0258942, 129.5988312
7: -80.7671127, 67.8903351, -64.1327133, 54.0716095, -134.8387146, 132.0230408
8: -96.3789444, 65.8685532, -76.0788040, 51.7647820, -148.1437225, 141.9473572
9: -73.6165619, 72.1117706, -58.4306831, 57.0006294, -130.6171875, 130.5424347

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4049566, upper bound: 197.4040017
time: 6.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4049566, upper bound: 197.4042846
time: 6.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -80.7031860, 64.1475143, -71.2988281, 56.5714684, -137.2746582, 135.4463501
1: -66.9119949, 56.8724823, -58.9950790, 50.2210693, -117.1330643, 115.8675613
2: -88.6478882, 58.0840797, -78.2026062, 51.4425201, -140.0904083, 136.2866821
3: -94.5149307, 49.6180534, -83.6586075, 43.8634796, -138.3783875, 133.2766571
4: -86.8864899, 66.8371811, -76.6757889, 58.9907455, -145.8772278, 143.5129547
5: -77.5939407, 60.4849434, -68.5557098, 53.4519424, -131.0458832, 129.0406342
6: -74.8100815, 71.2090073, -66.2247238, 62.8158684, -137.6259460, 137.4337311
7: -81.4398193, 68.4465485, -72.0952911, 60.6475906, -142.0874023, 140.5418396
8: -97.1832428, 66.4191284, -85.6167450, 58.2783546, -155.4615936, 152.0358124
9: -74.2284241, 72.7180023, -65.6587906, 64.1624985, -138.3909302, 138.3767700

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4054591, upper bound: 197.4049823
time: 8.48 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4054591, upper bound: 197.4061057
time: 9.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -86.7092133, 68.9003448, -64.2197647, 51.0040817, -137.7132874, 133.1201172
1: -72.0300369, 61.1328964, -53.1545982, 45.2902412, -117.3202591, 114.2874908
2: -95.2780991, 62.3980560, -70.4391861, 46.4416122, -141.7196808, 132.8372192
3: -101.5610733, 53.3298492, -75.3948669, 39.5263748, -141.0874481, 128.7247162
4: -93.3497086, 71.8128891, -69.0884781, 53.2651253, -146.6148071, 140.9013672
5: -83.3418884, 64.9677963, -61.8047142, 48.2426605, -131.5845490, 126.7724991
6: -80.3620453, 76.5467453, -59.7748184, 56.5934372, -136.9554749, 136.3215637
7: -87.5160065, 73.5404739, -65.0039978, 54.7991371, -142.3151398, 138.5444641
8: -104.4620667, 71.3897858, -77.1237946, 52.4790840, -156.9411469, 148.5135803
9: -79.8154602, 78.2409058, -59.2300949, 57.7926559, -137.6081238, 137.4710083

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4052687, upper bound: 197.4044115
time: 6.72 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4052687, upper bound: 197.4047016
time: 7.47 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -87.3752670, 69.4241486, -72.1585693, 57.2499733, -144.6252441, 141.5827179
1: -72.5820847, 61.5977211, -59.7260551, 50.8299408, -123.4120102, 121.3237762
2: -96.0093689, 62.8682785, -79.1513901, 52.0581474, -148.0675049, 142.0196381
3: -102.3392868, 53.7390404, -84.6666565, 44.3948631, -146.7341461, 138.4056854
4: -94.0621719, 72.3532715, -77.5997849, 59.7008934, -153.7630615, 149.9530640
5: -83.9776230, 65.4591446, -69.3774261, 54.0924072, -138.0700378, 134.8365784
6: -80.9678345, 77.1332092, -67.0160065, 63.5798798, -144.5477142, 144.1492157
7: -88.1834183, 74.0923157, -72.9633865, 61.3727074, -149.5561218, 147.0556946
8: -105.2595367, 71.9362488, -86.6569061, 58.9899521, -164.2494812, 158.5931549
9: -80.4225464, 78.8420486, -66.4556046, 64.9511642, -145.3736725, 145.2976379

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4057211, upper bound: 197.4052480
time: 7.01 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4057211, upper bound: 197.4065540
time: 6.41 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -80.9900284, 64.3731308, -80.9900284, 64.3731308, -145.3631287, 145.3631287
1: -67.1509247, 57.0728683, -67.1509247, 57.0728683, -124.2237701, 124.2237701
2: -88.9628677, 58.2864723, -88.9628677, 58.2864723, -147.2493286, 147.2493286
3: -94.8495255, 49.7938385, -94.8495255, 49.7938385, -144.6433716, 144.6433716
4: -87.1935120, 67.0708923, -87.1935120, 67.0708923, -154.2644043, 154.2644043
5: -77.8680878, 60.6965294, -77.8680878, 60.6965294, -138.5646210, 138.5646210
6: -75.0712967, 71.4620056, -75.0712967, 71.4620056, -146.5332947, 146.5332947
7: -81.7263260, 68.6844559, -81.7263260, 68.6844559, -150.4107819, 150.4107819
8: -97.5274811, 66.6552429, -97.5274811, 66.6552429, -164.1827087, 164.1827087
9: -74.4901352, 72.9769440, -74.4901352, 72.9769440, -147.4670715, 147.4670715

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4317255, upper bound: 197.4318445
time: 5.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4326171, upper bound: 197.4326221
time: 5.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -80.9900284, 64.3731308, -87.6675110, 69.6542511, -150.6442871, 152.0406342
1: -67.1509247, 57.0728683, -72.8256454, 61.8019753, -128.9528961, 129.8985138
2: -88.9628677, 58.2864723, -96.3304672, 63.0746346, -152.0375061, 154.6169434
3: -94.8495255, 49.7938385, -102.6804047, 53.9181709, -148.7676849, 152.4742432
4: -87.1935120, 67.0708923, -94.3752365, 72.5914841, -159.7849884, 161.4461365
5: -77.8680878, 60.6965294, -84.2571259, 65.6748962, -143.5429840, 144.9536591
6: -75.0712967, 71.4620056, -81.2340698, 77.3911514, -152.4624481, 152.6960754
7: -81.7263260, 68.6844559, -88.4754868, 74.3349457, -156.0612793, 157.1599274
8: -97.5274811, 66.6552429, -105.6106262, 72.1767273, -169.7041931, 172.2658539
9: -74.4901352, 72.9769440, -80.6891785, 79.1061096, -153.5962372, 153.6661224

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4317255, upper bound: 197.4320685
time: 5.86 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4326171, upper bound: 197.4328300
time: 5.22 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -87.6675110, 69.6542511, -80.9900284, 64.3731308, -152.0406342, 150.6442871
1: -72.8256454, 61.8019753, -67.1509247, 57.0728683, -129.8985138, 128.9528961
2: -96.3304672, 63.0746346, -88.9628677, 58.2864723, -154.6169434, 152.0375061
3: -102.6804047, 53.9181709, -94.8495255, 49.7938385, -152.4742432, 148.7676849
4: -94.3752365, 72.5914841, -87.1935120, 67.0708923, -161.4461365, 159.7849884
5: -84.2571259, 65.6748962, -77.8680878, 60.6965294, -144.9536591, 143.5429840
6: -81.2340698, 77.3911514, -75.0712967, 71.4620056, -152.6960754, 152.4624481
7: -88.4754868, 74.3349457, -81.7263260, 68.6844559, -157.1599274, 156.0612793
8: -105.6106262, 72.1767273, -97.5274811, 66.6552429, -172.2658539, 169.7041931
9: -80.6891785, 79.1061096, -74.4901352, 72.9769440, -153.6661224, 153.5962372

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4319492, upper bound: 197.4320645
time: 6.03 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4328332, upper bound: 197.4328319
time: 5.45 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -87.6675110, 69.6542511, -87.6675110, 69.6542511, -157.3217621, 157.3217621
1: -72.8256454, 61.8019753, -72.8256454, 61.8019753, -134.6276245, 134.6276245
2: -96.3304672, 63.0746346, -96.3304672, 63.0746346, -159.4051056, 159.4051056
3: -102.6804047, 53.9181709, -102.6804047, 53.9181709, -156.5985413, 156.5985413
4: -94.3752365, 72.5914841, -94.3752365, 72.5914841, -166.9667206, 166.9667206
5: -84.2571259, 65.6748962, -84.2571259, 65.6748962, -149.9320221, 149.9320221
6: -81.2340698, 77.3911514, -81.2340698, 77.3911514, -158.6252136, 158.6252136
7: -88.4754868, 74.3349457, -88.4754868, 74.3349457, -162.8104248, 162.8104248
8: -105.6106262, 72.1767273, -105.6106262, 72.1767273, -177.7873383, 177.7873383
9: -80.6891785, 79.1061096, -80.6891785, 79.1061096, -159.7952881, 159.7952881

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4319492, upper bound: 197.4325829
time: 5.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4328332, upper bound: 197.4333609
time: 5.33 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 12.45 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.3930257, upper bound: 197.3929668
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.3930257, upper bound: 197.3936674
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.3937216, upper bound: 197.3941758
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.3937216, upper bound: 197.3955896
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.3932323, upper bound: 197.3932323
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.3932323, upper bound: 197.3939268
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.3939268, upper bound: 197.3944233
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.3939268, upper bound: 197.3957764
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4043383, upper bound: 197.4054724
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4061244, upper bound: 197.4067806
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4043383, upper bound: 197.4056164
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4061244, upper bound: 197.4069456
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4042846, upper bound: 197.4054953
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4061057, upper bound: 197.4067875
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4042846, upper bound: 197.4057881
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4061057, upper bound: 197.4070641
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4049566, upper bound: 197.4040017
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4049566, upper bound: 197.4042846
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4054591, upper bound: 197.4049823
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4054591, upper bound: 197.4061057
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4052687, upper bound: 197.4044115
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4052687, upper bound: 197.4047016
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4057211, upper bound: 197.4052480
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4057211, upper bound: 197.4065540
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4317255, upper bound: 197.4318445
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4326171, upper bound: 197.4326221
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4317255, upper bound: 197.4320685
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4326171, upper bound: 197.4328300
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4319492, upper bound: 197.4320645
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4328332, upper bound: 197.4328319
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4319492, upper bound: 197.4325829
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.45
Output dim: 4, lower bound: -197.4328332, upper bound: 197.4333609

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -47.7533264, 38.0067062, -63.3569565, 50.3229637, -98.0762939, 101.3636551
1: -39.1894188, 33.6500435, -52.4208069, 44.6792297, -83.8686523, 86.0708466
2: -52.2731438, 34.6811256, -69.4866028, 45.8237076, -98.0968475, 104.1677170
3: -56.0520935, 29.3612862, -74.3834381, 38.9929352, -95.0450287, 103.7447205
4: -51.4136314, 39.7007751, -68.1608353, 52.5523834, -103.9660187, 107.8616028
5: -46.0653572, 35.9768867, -60.9801407, 47.5998383, -93.6651764, 96.9570160
6: -44.6461678, 41.9899292, -58.9809036, 55.8263893, -100.4725571, 100.9708328
7: -48.3966713, 40.9178696, -64.1327133, 54.0716095, -102.4682770, 105.0505829
8: -57.2190247, 38.8918228, -76.0788040, 51.7647820, -108.9838104, 114.9706192
9: -44.0193825, 42.7410431, -58.4306831, 57.0006294, -101.0200119, 101.1717224

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3928824, upper bound: 197.3928706
time: 5.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3928824, upper bound: 197.3929668
time: 5.01 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -55.2802963, 43.9292145, -63.3569565, 50.3229637, -105.6032486, 107.2861557
1: -45.3988686, 38.9006462, -52.4208069, 44.6792297, -90.0780945, 91.3214493
2: -60.5275192, 40.0008430, -69.4866028, 45.8237076, -106.3512268, 109.4874344
3: -64.8830032, 33.9735451, -74.3834381, 38.9929352, -103.8759384, 108.3569794
4: -59.4740372, 45.7926636, -68.1608353, 52.5523834, -112.0264206, 113.9534912
5: -53.2480888, 41.5344543, -60.9801407, 47.5998383, -100.8479233, 102.5145798
6: -51.5101814, 48.5928688, -58.9809036, 55.8263893, -107.3365707, 107.5737457
7: -55.9539528, 47.1704750, -64.1327133, 54.0716095, -110.0255585, 111.3031845
8: -66.2524796, 45.0326042, -76.0788040, 51.7647820, -118.0172577, 121.1113892
9: -50.8704987, 49.5251846, -58.4306831, 57.0006294, -107.8711243, 107.9558716

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3928824, upper bound: 197.3935919
time: 5.13 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3928824, upper bound: 197.3936674
time: 5.23 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -47.7533264, 38.0067062, -71.2988281, 56.5714684, -104.3247986, 109.3055344
1: -39.1894188, 33.6500435, -58.9950790, 50.2210693, -89.4104843, 92.6451263
2: -52.2731438, 34.6811256, -78.2026062, 51.4425201, -103.7156677, 112.8837204
3: -56.0520935, 29.3612862, -83.6586075, 43.8634796, -99.9155502, 113.0198898
4: -51.4136314, 39.7007751, -76.6757889, 58.9907455, -110.4043732, 116.3765564
5: -46.0653572, 35.9768867, -68.5557098, 53.4519424, -99.5173035, 104.5325928
6: -44.6461678, 41.9899292, -66.2247238, 62.8158684, -107.4620361, 108.2146530
7: -48.3966713, 40.9178696, -72.0952911, 60.6475906, -109.0442657, 113.0131607
8: -57.2190247, 38.8918228, -85.6167450, 58.2783546, -115.4973755, 124.5085602
9: -44.0193825, 42.7410431, -65.6587906, 64.1624985, -108.1818771, 108.3998337

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 168

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3928824, upper bound: 197.3941162
time: 5.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3928824, upper bound: 197.3941758
time: 5.44 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -55.2802963, 43.9292145, -71.2988281, 56.5714684, -111.8517532, 115.2280273
1: -45.3988686, 38.9006462, -58.9950790, 50.2210693, -95.6199341, 97.8957214
2: -60.5275192, 40.0008430, -78.2026062, 51.4425201, -111.9700394, 118.2034378
3: -64.8830032, 33.9735451, -83.6586075, 43.8634796, -108.7464676, 117.6321487
4: -59.4740372, 45.7926636, -76.6757889, 58.9907455, -118.4647827, 122.4684448
5: -53.2480888, 41.5344543, -68.5557098, 53.4519424, -106.7000275, 110.0901489
6: -51.5101814, 48.5928688, -66.2247238, 62.8158684, -114.3260498, 114.8175812
7: -55.9539528, 47.1704750, -72.0952911, 60.6475906, -116.6015472, 119.2657547
8: -66.2524796, 45.0326042, -85.6167450, 58.2783546, -124.5308380, 130.6493073
9: -50.8704987, 49.5251846, -65.6587906, 64.1624985, -115.0329971, 115.1839752

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 168

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3928824, upper bound: 197.3955751
time: 5.38 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3928824, upper bound: 197.3955892
time: 5.33 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -53.4165268, 42.4888916, -64.2197647, 51.0040817, -104.4206085, 106.7086563
1: -43.9720383, 37.6513290, -53.1545982, 45.2902412, -89.2622681, 90.8059235
2: -58.5177803, 38.7427025, -70.4391861, 46.4416122, -104.9593887, 109.1818771
3: -62.7286339, 32.8552017, -75.3948669, 39.5263748, -102.2550049, 108.2500687
4: -57.4898453, 44.3821526, -69.0884781, 53.2651253, -110.7549744, 113.4706039
5: -51.4907303, 40.2127876, -61.8047142, 48.2426605, -99.7333908, 102.0175018
6: -49.8724785, 46.9988976, -59.7748184, 56.5934372, -106.4659119, 106.7737045
7: -54.1294022, 45.7350616, -65.0039978, 54.7991371, -108.9285431, 110.7390594
8: -64.0754318, 43.5421753, -77.1237946, 52.4790840, -116.5545044, 120.6659698
9: -49.2860260, 47.9565163, -59.2300949, 57.7926559, -107.0786819, 107.1866150

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3929668, upper bound: 197.3930257
time: 5.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3929668, upper bound: 197.3932323
time: 5.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -61.1933899, 48.6001358, -64.2197647, 51.0040817, -112.1974716, 112.8199005
1: -50.4075966, 43.0807304, -53.1545982, 45.2902412, -95.6978302, 96.2353210
2: -67.0443573, 44.2436523, -70.4391861, 46.4416122, -113.4859619, 114.6828308
3: -71.8257980, 37.6242523, -75.3948669, 39.5263748, -111.3521576, 113.0191116
4: -65.8226929, 50.6755829, -69.0884781, 53.2651253, -119.0878067, 119.7640457
5: -58.9050522, 45.9473267, -61.8047142, 48.2426605, -107.1477127, 107.7520447
6: -56.9692802, 53.8328629, -59.7748184, 56.5934372, -113.5627060, 113.6076508
7: -61.9322777, 52.1849174, -65.0039978, 54.7991371, -116.7314148, 117.1889038
8: -73.4050674, 49.8967552, -77.1237946, 52.4790840, -125.8841476, 127.0205536
9: -56.3659248, 54.9629555, -59.2300949, 57.7926559, -114.1585846, 114.1930466

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3929668, upper bound: 197.3937216
time: 5.23 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3929668, upper bound: 197.3939268
time: 6.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -53.4165268, 42.4888916, -72.1585693, 57.2499733, -110.6665039, 114.6474609
1: -43.9720383, 37.6513290, -59.7260551, 50.8299408, -94.8019638, 97.3773804
2: -58.5177803, 38.7427025, -79.1513901, 52.0581474, -110.5759277, 117.8940887
3: -62.7286339, 32.8552017, -84.6666565, 44.3948631, -107.1234970, 117.5218582
4: -57.4898453, 44.3821526, -77.5997849, 59.7008934, -117.1907349, 121.9819183
5: -51.4907303, 40.2127876, -69.3774261, 54.0924072, -105.5831375, 109.5902100
6: -49.8724785, 46.9988976, -67.0160065, 63.5798798, -113.4523468, 114.0149078
7: -54.1294022, 45.7350616, -72.9633865, 61.3727074, -115.5020981, 118.6984482
8: -64.0754318, 43.5421753, -86.6569061, 58.9899521, -123.0653763, 130.1990814
9: -49.2860260, 47.9565163, -66.4556046, 64.9511642, -114.2371902, 114.4121246

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3929668, upper bound: 197.3942882
time: 5.50 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3929668, upper bound: 197.3944233
time: 5.41 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -61.1933899, 48.6001358, -72.1585693, 57.2499733, -118.4433594, 120.7587051
1: -50.4075966, 43.0807304, -59.7260551, 50.8299408, -101.2375259, 102.8067780
2: -67.0443573, 44.2436523, -79.1513901, 52.0581474, -119.1025085, 123.3950348
3: -71.8257980, 37.6242523, -84.6666565, 44.3948631, -116.2206421, 122.2909088
4: -65.8226929, 50.6755829, -77.5997849, 59.7008934, -125.5235672, 128.2753601
5: -58.9050522, 45.9473267, -69.3774261, 54.0924072, -112.9974594, 115.3247528
6: -56.9692802, 53.8328629, -67.0160065, 63.5798798, -120.5491333, 120.8488541
7: -61.9322777, 52.1849174, -72.9633865, 61.3727074, -123.3049774, 125.1482925
8: -73.4050674, 49.8967552, -86.6569061, 58.9899521, -132.3949890, 136.5536499
9: -56.3659248, 54.9629555, -66.4556046, 64.9511642, -121.3170929, 121.4185638

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3929668, upper bound: 197.3956681
time: 5.22 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3929668, upper bound: 197.3957753
time: 4.92 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -47.7533264, 38.0067062, -80.0319366, 63.6196327, -111.3729553, 118.0386429
1: -39.1894188, 33.6500435, -66.3554535, 56.4039536, -95.5933609, 100.0054932
2: -52.2731438, 34.6811256, -87.9108963, 57.6097527, -109.8828964, 122.5920181
3: -56.0520935, 29.3612862, -93.7304688, 49.2056313, -105.2577057, 123.0917511
4: -51.4136314, 39.7007751, -86.1683121, 66.2925720, -117.7062073, 125.8690796
5: -46.0653572, 35.9768867, -76.9532928, 59.9896011, -106.0549622, 112.9301758
6: -44.6461678, 41.9899292, -74.1995010, 70.6179428, -115.2641068, 116.1894302
7: -48.3966713, 40.9178696, -80.7671127, 67.8903351, -116.2870026, 121.6849823
8: -57.2190247, 38.8918228, -96.3789444, 65.8685532, -123.0875778, 135.2707520
9: -44.0193825, 42.7410431, -73.6165619, 72.1117706, -116.1311340, 116.3575974

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4040253, upper bound: 197.4049522
time: 7.29 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4040253, upper bound: 197.4054724
time: 7.01 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -55.2802963, 43.9292145, -80.7031860, 64.1475143, -119.4278030, 124.6323929
1: -45.3988686, 38.9006462, -66.9119949, 56.8724823, -102.2713470, 105.8126373
2: -60.5275192, 40.0008430, -88.6478882, 58.0840797, -118.6116028, 128.6487274
3: -64.8830032, 33.9735451, -94.5149307, 49.6180534, -114.5010376, 128.4884644
4: -59.4740372, 45.7926636, -86.8864899, 66.8371811, -126.3112183, 132.6791534
5: -53.2480888, 41.5344543, -77.5939407, 60.4849434, -113.7330322, 119.1283951
6: -51.5101814, 48.5928688, -74.8100815, 71.2090073, -122.7191849, 123.4029388
7: -55.9539528, 47.1704750, -81.4398193, 68.4465485, -124.4004974, 128.6102905
8: -66.2524796, 45.0326042, -97.1832428, 66.4191284, -132.6715698, 142.2158203
9: -50.8704987, 49.5251846, -74.2284241, 72.7180023, -123.5885010, 123.7536087

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4050012, upper bound: 197.4054617
time: 6.50 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4050012, upper bound: 197.4054617
time: 6.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -47.7533264, 38.0067062, -86.7092133, 68.9003448, -116.6536713, 124.7159195
1: -39.1894188, 33.6500435, -72.0300369, 61.1328964, -100.3222961, 105.6800690
2: -52.2731438, 34.6811256, -95.2780991, 62.3980560, -114.6712036, 129.9592133
3: -56.0520935, 29.3612862, -101.5610733, 53.3298492, -109.3819275, 130.9223480
4: -51.4136314, 39.7007751, -93.3497086, 71.8128891, -123.2265167, 133.0504761
5: -46.0653572, 35.9768867, -83.3418884, 64.9677963, -111.0331421, 119.3187714
6: -44.6461678, 41.9899292, -80.3620453, 76.5467453, -121.1929092, 122.3519745
7: -48.3966713, 40.9178696, -87.5160065, 73.5404739, -121.9371490, 128.4338684
8: -57.2190247, 38.8918228, -104.4620667, 71.3897858, -128.6088104, 143.3538818
9: -44.0193825, 42.7410431, -79.8154602, 78.2409058, -122.2602844, 122.5565033

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4042217, upper bound: 197.4050633
time: 7.97 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4042217, upper bound: 197.4056164
time: 7.93 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -55.2802963, 43.9292145, -87.3752670, 69.4241486, -124.7044373, 131.3044739
1: -45.3988686, 38.9006462, -72.5820847, 61.5977211, -106.9965897, 111.4827271
2: -60.5275192, 40.0008430, -96.0093689, 62.8682785, -123.3957977, 136.0102081
3: -64.8830032, 33.9735451, -102.3392868, 53.7390404, -118.6220322, 136.3128357
4: -59.4740372, 45.7926636, -94.0621719, 72.3532715, -131.8273010, 139.8548279
5: -53.2480888, 41.5344543, -83.9776230, 65.4591446, -118.7072296, 125.5120773
6: -51.5101814, 48.5928688, -80.9678345, 77.1332092, -128.6433868, 129.5606995
7: -55.9539528, 47.1704750, -88.1834183, 74.0923157, -130.0462341, 135.3538971
8: -66.2524796, 45.0326042, -105.2595367, 71.9362488, -138.1887207, 150.2921295
9: -50.8704987, 49.5251846, -80.4225464, 78.8420486, -129.7125549, 129.9477234

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4051558, upper bound: 197.4055875
time: 5.49 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4051558, upper bound: 197.4069456
time: 6.13 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 13.17 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.3928824, upper bound: 197.3928706
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.3928824, upper bound: 197.3929668
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.3928824, upper bound: 197.3935919
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.3928824, upper bound: 197.3936674
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.3928824, upper bound: 197.3941162
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.3928824, upper bound: 197.3941758
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.3928824, upper bound: 197.3955751
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.3928824, upper bound: 197.3955892
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.3929668, upper bound: 197.3930257
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.3929668, upper bound: 197.3932323
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.3929668, upper bound: 197.3937216
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.3929668, upper bound: 197.3939268
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.3929668, upper bound: 197.3942882
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.3929668, upper bound: 197.3944233
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.3929668, upper bound: 197.3956681
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.3929668, upper bound: 197.3957753
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.4040253, upper bound: 197.4049522
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.4040253, upper bound: 197.4054724
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.4050012, upper bound: 197.4054617
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.4050012, upper bound: 197.4054617
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.4042217, upper bound: 197.4050633
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.4042217, upper bound: 197.4056164
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.4051558, upper bound: 197.4055875
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.17
Output dim: 4, lower bound: -197.4051558, upper bound: 197.4069456
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4042846, upper bound: 197.4054953
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4061057, upper bound: 197.4067875
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4042846, upper bound: 197.4057881
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4061057, upper bound: 197.4070641
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4049566, upper bound: 197.4040017
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4049566, upper bound: 197.4042846
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4054591, upper bound: 197.4049823
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4054591, upper bound: 197.4061057
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4052687, upper bound: 197.4044115
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4052687, upper bound: 197.4047016
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4057211, upper bound: 197.4052480
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4057211, upper bound: 197.4065540
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4317255, upper bound: 197.4318445
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4326171, upper bound: 197.4326221
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4317255, upper bound: 197.4320685
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4326171, upper bound: 197.4328300
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4319492, upper bound: 197.4320645
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4328332, upper bound: 197.4328319
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4319492, upper bound: 197.4325829
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.17
Output dim: 4, lower bound: -197.4328332, upper bound: 197.4333609
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=198.953369140625
rel_dist={4: [-197.44087218970873, 197.4408721892934]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4093851, upper bound: 197.4098506
time: 8.24 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4388501, upper bound: 197.4388501
time: 4.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.11 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.11
Output dim: 4, lower bound: -197.4093851, upper bound: 197.4098506
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.11
Output dim: 4, lower bound: -197.4388501, upper bound: 197.4388501

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -79.9444504, 63.3750114, -93.6110077, 74.2906799, -154.2351227, 156.9860229
1: -66.2200775, 56.2719460, -77.8618393, 65.9795837, -132.1996613, 134.1337891
2: -87.7009277, 57.5595970, -102.8611526, 67.3055725, -155.0064850, 160.4207458
3: -93.7524338, 49.1665268, -109.6692657, 57.5932503, -151.3456879, 158.8357544
4: -85.9368134, 66.0450745, -100.7016296, 77.4250488, -163.3618622, 166.7467041
5: -76.8184204, 59.8392181, -89.9155273, 70.0633163, -146.8817291, 149.7547302
6: -74.1077042, 70.4479294, -86.6628723, 82.6478958, -156.7556000, 157.1108093
7: -80.7441788, 67.8362045, -94.4707870, 79.3106003, -160.0547485, 162.3069611
8: -96.0069962, 65.3933868, -112.6965866, 76.9817963, -172.9888000, 178.0899506
9: -73.5597076, 71.9806671, -86.1536560, 84.4908371, -158.0505066, 158.1343231

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4034076, upper bound: 197.4040576
time: 8.21 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4037714, upper bound: 197.4043222
time: 7.07 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -99.1355057, 78.7032318, -103.5517502, 82.2159271, -181.3513794, 182.2549591
1: -82.5673141, 69.9038849, -86.3098831, 73.0283813, -155.5956573, 156.2137604
2: -108.9954758, 71.2377625, -113.8802032, 74.3851089, -183.3805847, 185.1179352
3: -116.1185074, 61.0004578, -121.2779465, 63.7236328, -179.8421326, 182.2783813
4: -106.6865845, 82.0365067, -111.4516983, 85.6835556, -192.3701477, 193.4881592
5: -95.2138977, 74.1955032, -99.4504166, 77.4904404, -172.7043152, 173.6459198
6: -91.7468567, 87.5859756, -95.8004303, 91.5136414, -183.2604980, 183.3863983
7: -100.0214844, 83.9518967, -104.4575272, 87.6625214, -187.6839905, 188.4094238
8: -119.4582748, 81.6738663, -124.8128204, 85.3636856, -204.8219452, 206.4866791
9: -91.2429123, 89.5477371, -95.2969131, 93.5584030, -184.8013000, 184.8446350

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4098506, upper bound: 197.4093851
time: 7.36 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4098506, upper bound: 197.4388501
time: 7.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 15.79 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 15.79
Output dim: 4, lower bound: -197.4034076, upper bound: 197.4040576
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 15.79
Output dim: 4, lower bound: -197.4037714, upper bound: 197.4043222
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 15.79
Output dim: 4, lower bound: -197.4098506, upper bound: 197.4093851
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 15.79
Output dim: 4, lower bound: -197.4098506, upper bound: 197.4388501

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -73.0316467, 57.9180489, -75.6849442, 60.1346016, -133.1662445, 133.6029663
1: -60.3441200, 51.3801651, -62.6311302, 53.2998581, -113.6439743, 114.0112839
2: -80.0673218, 52.6203270, -83.0660782, 54.5087700, -134.5760345, 135.6864014
3: -85.6491470, 44.8958778, -88.6635284, 46.5233040, -132.1724548, 133.5594025
4: -78.5099258, 60.3406448, -81.4453278, 62.6406708, -141.1506042, 141.7859497
5: -70.2130051, 54.6945190, -72.7830734, 56.7273750, -126.9403839, 127.4775925
6: -67.7568207, 64.3030548, -70.1905365, 66.7155838, -134.4724121, 134.4935760
7: -73.7721176, 62.0215416, -76.3979111, 64.2283859, -138.0005035, 138.4194183
8: -87.6442566, 59.6669273, -91.0238953, 62.1415443, -149.7857971, 150.6908112
9: -67.1719589, 65.6571884, -69.6005783, 68.1161194, -135.2880859, 135.2577667

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 168

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3990703, upper bound: 197.3998343
time: 9.62 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4012846, upper bound: 197.4017169
time: 7.08 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -75.3317871, 59.7350540, -82.0703354, 65.1801453, -140.5119324, 141.8053894
1: -62.3005295, 53.0110550, -68.0527954, 57.8200951, -120.1206207, 121.0638504
2: -82.6068497, 54.2722511, -90.1104736, 59.0871429, -141.6939697, 144.3827209
3: -88.3486023, 46.3182869, -96.1490173, 50.4647293, -138.8133240, 142.4672852
4: -80.9839172, 62.2458687, -88.3116684, 67.9164124, -148.9003296, 150.5575256
5: -72.4125366, 56.4119797, -78.8885956, 61.4874153, -133.8999481, 135.3005676
6: -69.8801880, 66.3476715, -76.0839462, 72.3854599, -142.2656250, 142.4316101
7: -76.1012115, 63.9697990, -82.8512955, 69.6323013, -145.7334900, 146.8210907
8: -90.4331894, 61.5680618, -98.7549820, 67.4131393, -157.8463135, 160.3230438
9: -69.3129883, 67.7797928, -75.5298309, 73.9788437, -143.2918091, 143.3096313

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3995486, upper bound: 197.4002069
time: 8.77 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4017135, upper bound: 197.4020573
time: 7.68 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -99.1355057, 78.7032318, -79.9444504, 63.3750114, -162.5104828, 158.6476593
1: -82.5673141, 69.9038849, -66.2200775, 56.2719460, -138.8392639, 136.1239624
2: -108.9954758, 71.2377625, -87.7009277, 57.5595970, -166.5550690, 158.9386597
3: -116.1185074, 61.0004578, -93.7524338, 49.1665268, -165.2850037, 154.7528839
4: -106.6865845, 82.0365067, -85.9368134, 66.0450745, -172.7316589, 167.9733124
5: -95.2138977, 74.1955032, -76.8184204, 59.8392181, -155.0531158, 151.0139160
6: -91.7468567, 87.5859756, -74.1077042, 70.4479294, -162.1947937, 161.6936798
7: -100.0214844, 83.9518967, -80.7441788, 67.8362045, -167.8576660, 164.6960754
8: -119.4582748, 81.6738663, -96.0069962, 65.3933868, -184.8516235, 177.6808624
9: -91.2429123, 89.5477371, -73.5597076, 71.9806671, -163.2235718, 163.1074371

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4040576, upper bound: 197.4034076
time: 6.69 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4043222, upper bound: 197.4037714
time: 8.61 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -99.1355057, 78.7032318, -99.1355057, 78.7032318, -177.8386688, 177.8386688
1: -82.5673141, 69.9038849, -82.5673141, 69.9038849, -152.4711609, 152.4711609
2: -108.9954758, 71.2377625, -108.9954758, 71.2377625, -180.2332306, 180.2332306
3: -116.1185074, 61.0004578, -116.1185074, 61.0004578, -177.1189117, 177.1189117
4: -106.6865845, 82.0365067, -106.6865845, 82.0365067, -188.7230530, 188.7230530
5: -95.2138977, 74.1955032, -95.2138977, 74.1955032, -169.4093933, 169.4093933
6: -91.7468567, 87.5859756, -91.7468567, 87.5859756, -179.3328247, 179.3328247
7: -100.0214844, 83.9518967, -100.0214844, 83.9518967, -183.9733887, 183.9733887
8: -119.4582748, 81.6738663, -119.4582748, 81.6738663, -201.1321259, 201.1321259
9: -91.2429123, 89.5477371, -91.2429123, 89.5477371, -180.7906494, 180.7906494

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4040576, upper bound: 197.4349020
time: 6.92 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4043222, upper bound: 197.4037714
time: 8.96 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 17.32 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.32
Output dim: 4, lower bound: -197.3990703, upper bound: 197.3998343
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.32
Output dim: 4, lower bound: -197.4012846, upper bound: 197.4017169
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.32
Output dim: 4, lower bound: -197.3995486, upper bound: 197.4002069
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.32
Output dim: 4, lower bound: -197.4017135, upper bound: 197.4020573
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.32
Output dim: 4, lower bound: -197.4040576, upper bound: 197.4034076
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.32
Output dim: 4, lower bound: -197.4043222, upper bound: 197.4037714
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.32
Output dim: 4, lower bound: -197.4040576, upper bound: 197.4349020
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.32
Output dim: 4, lower bound: -197.4043222, upper bound: 197.4037714

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -57.9427338, 46.0502853, -70.4626999, 56.0282097, -113.9709473, 116.5129852
1: -47.8214684, 40.8501358, -58.2938728, 49.6530418, -97.4745102, 99.1440125
2: -63.5096283, 41.9536591, -77.3342590, 50.8208275, -114.3304596, 119.2879181
3: -68.0391769, 35.6473465, -82.5662689, 43.3153915, -111.3545609, 118.2135925
4: -62.3432198, 48.0855103, -75.8566132, 58.3983650, -120.7415848, 123.9421234
5: -55.8076591, 43.5713387, -67.7981415, 52.8739662, -108.6816254, 111.3694687
6: -54.0077972, 51.0143890, -65.4401321, 62.1155243, -116.1233063, 116.4545212
7: -58.6748962, 49.5177422, -71.1705475, 59.9031448, -118.5780334, 120.6882858
8: -69.5280075, 47.2781334, -84.7654190, 57.8496094, -127.3776169, 132.0435486
9: -53.4291344, 52.0453949, -64.8390274, 63.3995895, -116.8287201, 116.8844223

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3932489, upper bound: 197.3936768
time: 6.60 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3932489, upper bound: 197.3998343
time: 5.83 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -65.8794403, 52.2954102, -72.9649277, 57.9946861, -123.8741226, 125.2603378
1: -54.3890953, 46.3860855, -60.3662300, 51.4006844, -105.7897797, 106.7523193
2: -72.2206726, 47.5700035, -80.0800705, 52.5894127, -124.8100891, 127.6500702
3: -77.3069305, 40.5152245, -85.4911652, 44.8560715, -122.1630020, 126.0063934
4: -70.8522568, 54.5207787, -78.5336304, 60.4258194, -131.2780762, 133.0544128
5: -63.3780975, 49.4197197, -70.1838226, 54.7208481, -118.0989304, 119.6035461
6: -61.2465439, 57.9995079, -67.7139282, 64.3172913, -125.5638351, 125.7134399
7: -66.6315079, 56.0905304, -73.6806335, 61.9724655, -128.6039581, 129.7711639
8: -79.0618134, 53.7890968, -87.7609787, 59.9042397, -138.9660492, 141.5500793
9: -60.6512718, 59.2058563, -67.1188202, 65.6608353, -126.3121033, 126.3246765

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3955396, upper bound: 197.3955912
time: 6.63 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3955396, upper bound: 197.4017169
time: 7.01 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -60.2420197, 47.8670578, -76.8394470, 61.0653992, -121.3074112, 124.7065048
1: -49.7753067, 42.4780426, -63.7083130, 54.1665840, -103.9418945, 106.1863556
2: -66.0479279, 43.6061211, -84.3665390, 55.3923912, -121.4403076, 127.9726410
3: -70.7364349, 37.0704155, -90.0398178, 47.2520943, -117.9885254, 127.1102295
4: -64.8158646, 49.9908714, -82.7122116, 63.6658783, -128.4817047, 132.7030792
5: -58.0063400, 45.2878685, -73.8942566, 57.6271324, -115.6334686, 119.1821289
6: -56.1291389, 53.0588417, -71.3235855, 67.7776871, -123.9068146, 124.3824310
7: -61.0022278, 51.4662247, -77.6131973, 65.2984772, -126.3007050, 129.0794067
8: -72.3168259, 49.1797638, -92.4851456, 63.1140747, -135.4308777, 141.6648865
9: -55.5688286, 54.1698914, -70.7597351, 69.2548294, -124.8236542, 124.9296265

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3935548, upper bound: 197.3939135
time: 5.94 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3935548, upper bound: 197.4002070
time: 5.96 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -68.1773148, 54.1091690, -79.3142090, 63.0113335, -131.1886292, 133.4233704
1: -56.3429832, 48.0154800, -65.7577209, 55.8949928, -112.2379684, 113.7732010
2: -74.7554169, 49.2199211, -87.0832596, 57.1420746, -131.8974915, 136.3031769
3: -80.0022659, 41.9368668, -92.9332886, 48.7755280, -128.7778015, 134.8701477
4: -73.3227768, 56.4227982, -85.3604202, 65.6712646, -138.9940491, 141.7832031
5: -65.5744629, 51.1348572, -76.2549057, 59.4534836, -125.0279465, 127.3897629
6: -63.3671684, 60.0410423, -73.5738220, 69.9542389, -133.3214111, 133.6148529
7: -68.9566498, 58.0363235, -80.0968933, 67.3466797, -136.3033295, 138.1331940
8: -81.8438416, 55.6883812, -95.4462814, 65.1454468, -146.9892883, 151.1346588
9: -62.7911453, 61.3247604, -73.0153503, 71.4899216, -134.2810516, 134.3400879

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3556151, upper bound: 197.3552974
time: 7.27 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3998998, upper bound: 197.4002612
time: 7.51 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -80.9900284, 64.3731308, -73.0316467, 57.9180489, -138.9080658, 137.4047546
1: -67.1509247, 57.0728683, -60.3441200, 51.3801651, -118.5310745, 117.4169922
2: -88.9628677, 58.2864723, -80.0673218, 52.6203270, -141.5831909, 138.3537445
3: -94.8495255, 49.7938385, -85.6491470, 44.8958778, -139.7454071, 135.4429932
4: -87.1935120, 67.0708923, -78.5099258, 60.3406448, -147.5341492, 145.5808105
5: -77.8680878, 60.6965294, -70.2130051, 54.6945190, -132.5626068, 130.9095306
6: -75.0712967, 71.4620056, -67.7568207, 64.3030548, -139.3743439, 139.2188263
7: -81.7263260, 68.6844559, -73.7721176, 62.0215416, -143.7478485, 142.4565735
8: -97.5274811, 66.6552429, -87.6442566, 59.6669273, -157.1943817, 154.2994995
9: -74.4901352, 72.9769440, -67.1719589, 65.6571884, -140.1473236, 140.1488953

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 168

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3998342, upper bound: 197.3990703
time: 8.89 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4017169, upper bound: 197.4012846
time: 6.50 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -87.6675110, 69.6542511, -75.3317871, 59.7350540, -147.4025574, 144.9860382
1: -72.8256454, 61.8019753, -62.3005295, 53.0110550, -125.8367004, 124.1025009
2: -96.3304672, 63.0746346, -82.6068497, 54.2722511, -150.6027222, 145.6814575
3: -102.6804047, 53.9181709, -88.3486023, 46.3182869, -148.9986725, 142.2667542
4: -94.3752365, 72.5914841, -80.9839172, 62.2458687, -156.6211090, 153.5753937
5: -84.2571259, 65.6748962, -72.4125366, 56.4119797, -140.6690979, 138.0874329
6: -81.2340698, 77.3911514, -69.8801880, 66.3476715, -147.5817413, 147.2713165
7: -88.4754868, 74.3349457, -76.1012115, 63.9697990, -152.4452820, 150.4361572
8: -105.6106262, 72.1767273, -90.4331894, 61.5680618, -167.1786652, 162.6099091
9: -80.6891785, 79.1061096, -69.3129883, 67.7797928, -148.4689636, 148.4190826

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4002069, upper bound: 197.3995486
time: 7.34 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4020573, upper bound: 197.4017135
time: 7.28 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -80.9900284, 64.3731308, -92.0331955, 73.0945358, -154.0845490, 156.4063110
1: -67.1509247, 57.0728683, -76.5306625, 64.8809509, -132.0318756, 133.6035309
2: -88.9628677, 58.2864723, -101.1530609, 66.1669846, -155.1298523, 159.4395294
3: -94.8495255, 49.7938385, -107.7886047, 56.6104317, -151.4599609, 157.5824432
4: -87.1935120, 67.0708923, -99.0544281, 76.1752396, -163.3687439, 166.1253204
5: -77.8680878, 60.6965294, -88.4246063, 68.9097595, -146.7778473, 149.1211395
6: -75.0712967, 71.4620056, -85.2191925, 81.2735596, -156.3448486, 156.6811829
7: -81.7263260, 68.6844559, -92.8564529, 77.9755554, -159.7018738, 161.5409088
8: -97.5274811, 66.6552429, -110.8734207, 75.7917480, -173.3192291, 177.5286560
9: -74.4901352, 72.9769440, -84.6815414, 83.0557251, -157.5458527, 157.6584778

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4317643, upper bound: 197.4316913
time: 6.19 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4326845, upper bound: 197.4326838
time: 6.95 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -87.6675110, 69.6542511, -94.5711670, 75.1020737, -162.7695923, 164.2254028
1: -72.8256454, 61.8019753, -78.6913071, 66.6797943, -139.5054321, 140.4932556
2: -96.3304672, 63.0746346, -103.9555054, 67.9882889, -164.3187561, 167.0301361
3: -102.6804047, 53.9181709, -110.7691803, 58.1814957, -160.8618774, 164.6873322
4: -94.3752365, 72.5914841, -101.7866516, 78.2773743, -172.6526031, 174.3781281
5: -84.2571259, 65.6748962, -90.8530884, 70.8041534, -155.0612793, 156.5279846
6: -81.2340698, 77.3911514, -87.5619812, 83.5290527, -164.7631226, 164.9531250
7: -88.4754868, 74.3349457, -95.4252090, 80.1242294, -168.5997162, 169.7601624
8: -105.6106262, 72.1767273, -113.9477844, 77.8950272, -183.5056305, 186.1245117
9: -80.6891785, 79.1061096, -87.0421753, 85.3920898, -166.0812683, 166.1482849

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4323521, upper bound: 197.4322804
time: 7.04 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4332869, upper bound: 197.4332890
time: 5.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 14.27 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.27
Output dim: 4, lower bound: -197.3932489, upper bound: 197.3936768
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.27
Output dim: 4, lower bound: -197.3932489, upper bound: 197.3998343
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.27
Output dim: 4, lower bound: -197.3955396, upper bound: 197.3955912
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.27
Output dim: 4, lower bound: -197.3955396, upper bound: 197.4017169
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.27
Output dim: 4, lower bound: -197.3935548, upper bound: 197.3939135
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.27
Output dim: 4, lower bound: -197.3935548, upper bound: 197.4002070
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.27
Output dim: 4, lower bound: -197.3556151, upper bound: 197.3552974
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.27
Output dim: 4, lower bound: -197.3998998, upper bound: 197.4002612
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.27
Output dim: 4, lower bound: -197.3998342, upper bound: 197.3990703
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.27
Output dim: 4, lower bound: -197.4017169, upper bound: 197.4012846
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.27
Output dim: 4, lower bound: -197.4002069, upper bound: 197.3995486
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.27
Output dim: 4, lower bound: -197.4020573, upper bound: 197.4017135
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.27
Output dim: 4, lower bound: -197.4317643, upper bound: 197.4316913
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.27
Output dim: 4, lower bound: -197.4326845, upper bound: 197.4326838
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.27
Output dim: 4, lower bound: -197.4323521, upper bound: 197.4322804
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.27
Output dim: 4, lower bound: -197.4332869, upper bound: 197.4332890

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -57.9427338, 46.0502853, -56.8708534, 45.1799011, -103.1226349, 102.9211426
1: -47.8214684, 40.8501358, -46.7256660, 40.0073929, -87.8288574, 87.5758057
2: -63.5096283, 41.9536591, -62.2637253, 41.1164856, -104.6261063, 104.2173843
3: -68.0391769, 35.6473465, -66.7398453, 34.9410629, -102.9802170, 102.3871689
4: -62.3432198, 48.0855103, -61.1706085, 47.0911636, -109.4343872, 109.2561111
5: -55.8076591, 43.5713387, -54.7701302, 42.7085228, -98.5161819, 98.3414688
6: -54.0077972, 51.0143890, -52.9590340, 49.9874458, -103.9952393, 103.9734192
7: -58.6748962, 49.5177422, -57.5324364, 48.4921188, -107.1670151, 107.0501633
8: -69.5280075, 47.2781334, -68.1527634, 46.3337097, -115.8617172, 115.4308853
9: -53.4291344, 52.0453949, -52.3251114, 50.9505348, -104.3796692, 104.3704987

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3435790, upper bound: 197.3448669
time: 8.47 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3914039, upper bound: 197.3917330
time: 7.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -57.9427338, 46.0502853, -75.8119965, 60.2978096, -118.2405396, 121.8622818
1: -47.8214684, 40.8501358, -62.8509064, 53.4563637, -101.2778244, 103.7010422
2: -63.5096283, 41.9536591, -83.2740860, 54.6312675, -118.1408997, 125.2277451
3: -68.0391769, 35.6473465, -88.8035431, 46.6117477, -114.6509094, 124.4508667
4: -62.3432198, 48.0855103, -81.6525497, 62.8625793, -125.2057953, 129.7380524
5: -55.8076591, 43.5713387, -72.9256134, 56.8753815, -112.6830292, 116.4969406
6: -54.0077972, 51.0143890, -70.3589935, 66.8992462, -120.9070435, 121.3733826
7: -58.6748962, 49.5177422, -76.5403214, 64.3945618, -123.0694504, 126.0580597
8: -69.5280075, 47.2781334, -91.3225021, 62.3986130, -131.9265747, 138.6006317
9: -53.4291344, 52.0453949, -69.7656937, 68.2994537, -121.7285843, 121.8110886

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3435790, upper bound: 197.3509764
time: 7.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3914039, upper bound: 197.3917330
time: 7.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -65.8794403, 52.2954102, -59.3823814, 47.1555328, -113.0349731, 111.6777725
1: -54.3890953, 46.3860855, -48.8047943, 41.7608223, -96.1499176, 95.1908722
2: -72.2206726, 47.5700035, -65.0233459, 42.8941574, -115.1148224, 112.5933533
3: -77.3069305, 40.5152245, -69.6765671, 36.4887199, -113.7956390, 110.1917877
4: -70.8522568, 54.5207787, -63.8604088, 49.1275330, -119.9797897, 118.3811646
5: -63.3780975, 49.4197197, -57.1678123, 44.5630150, -107.9411163, 106.5875320
6: -61.2465439, 57.9995079, -55.2435722, 52.1976051, -113.4441528, 113.2430801
7: -66.6315079, 56.0905304, -60.0515900, 50.5713768, -117.2028809, 116.1421051
8: -79.0618134, 53.7890968, -71.1646042, 48.4001923, -127.4620056, 124.9536896
9: -60.6512718, 59.2058563, -54.6130562, 53.2264557, -113.8777313, 113.8189087

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3445014, upper bound: 197.3455256
time: 7.98 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3935833, upper bound: 197.3936362
time: 6.52 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -65.8794403, 52.2954102, -78.2111206, 62.1873703, -128.0668030, 130.5065308
1: -54.3890953, 46.3860855, -64.8361664, 55.1317062, -109.5207901, 111.2222519
2: -72.2206726, 47.5700035, -85.9116669, 56.3260231, -128.5466919, 133.4816742
3: -77.3069305, 40.5152245, -91.6079941, 48.0906143, -125.3975372, 132.1232147
4: -70.8522568, 54.5207787, -84.2190933, 64.8059158, -135.6581573, 138.7398529
5: -63.3780975, 49.4197197, -75.2123718, 58.6466026, -122.0246811, 124.6320953
6: -61.2465439, 57.9995079, -72.5405655, 69.0117416, -130.2582855, 130.5400696
7: -66.6315079, 56.0905304, -78.9501953, 66.3804626, -133.0119629, 135.0407257
8: -79.0618134, 53.7890968, -94.1931534, 64.3665237, -143.4283142, 147.9822235
9: -60.6512718, 59.2058563, -71.9540558, 70.4683914, -131.1196442, 131.1599121

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3445014, upper bound: 197.3455256
time: 7.14 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3935833, upper bound: 197.3936362
time: 6.54 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -60.2420197, 47.8670578, -62.8464165, 49.9019470, -110.1439514, 110.7134705
1: -49.7753067, 42.4780426, -51.7920532, 44.2308540, -94.0061569, 94.2700958
2: -66.0479279, 43.6061211, -68.8544846, 45.4061241, -111.4540558, 112.4606018
3: -70.7364349, 37.0704155, -73.7485275, 38.6310387, -109.3674545, 110.8189392
4: -64.8158646, 49.9908714, -67.5910187, 52.0301514, -116.8460007, 117.5818787
5: -58.0063400, 45.2878685, -60.4884338, 47.1658325, -105.1721725, 105.7763062
6: -56.1291389, 53.0588417, -58.4762383, 55.2907486, -111.4198914, 111.5350800
7: -61.0022278, 51.4662247, -63.5737572, 53.5554466, -114.5576782, 115.0399551
8: -72.3168259, 49.1797638, -75.3903580, 51.2594261, -123.5762482, 124.5701218
9: -55.5688286, 54.1698914, -57.8748322, 56.4527016, -112.0215302, 112.0447083

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 140

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3446313, upper bound: 197.3458486
time: 7.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3916919, upper bound: 197.3919603
time: 6.20 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -60.2420197, 47.8670578, -82.5053101, 65.5929260, -125.8349457, 130.3723755
1: -49.7753067, 42.4780426, -68.5392532, 58.1979141, -107.9732132, 111.0172958
2: -66.0479279, 43.6061211, -90.6620712, 59.4303551, -125.4782867, 134.2681885
3: -70.7364349, 37.0704155, -96.6505051, 50.7477188, -121.4841537, 133.7209167
4: -64.8158646, 49.9908714, -88.8516541, 68.3969727, -133.2128296, 138.8425140
5: -58.0063400, 45.2878685, -79.3276138, 61.8655396, -119.8718796, 124.6154785
6: -56.1291389, 53.0588417, -76.5364380, 72.8436508, -128.9727631, 129.5952759
7: -61.0022278, 51.4662247, -83.3060150, 70.0573196, -131.0595398, 134.7722015
8: -72.3168259, 49.1797638, -99.4258194, 67.9357758, -140.2525787, 148.6055756
9: -55.5688286, 54.1698914, -75.9818802, 74.4452362, -130.0140381, 130.1517639

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3446313, upper bound: 197.3517804
time: 7.13 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3916919, upper bound: 197.3984237
time: 6.23 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -62.0469666, 49.2285309, -67.0202026, 53.1995049, -115.2464752, 116.2487259
1: -51.0891342, 43.6434135, -55.0931129, 47.0479889, -98.1371231, 98.7365265
2: -67.9903717, 44.8529129, -73.4976349, 48.3682632, -116.3586349, 118.3505478
3: -72.8682632, 38.1401863, -78.6756744, 41.1268463, -113.9951019, 116.8158569
4: -66.6970749, 51.3261032, -72.0598450, 55.3605576, -122.0576324, 123.3859482
5: -59.6983490, 46.5770988, -64.4900131, 50.2861137, -109.9844666, 111.0671082
6: -57.7032280, 54.5625877, -62.2053375, 58.9094963, -116.6127167, 116.7679291
7: -62.8005638, 52.8869438, -67.7314758, 57.0015945, -119.8021469, 120.6184158
8: -74.3370590, 50.4757309, -80.3169785, 54.5408249, -128.8778839, 130.7927094
9: -57.1556816, 55.7152481, -61.6664352, 60.1348381, -117.2905197, 117.3816757

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2743508, upper bound: 197.2751606
time: 9.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3429292, upper bound: 197.3423829
time: 6.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3552385, upper bound: 197.3549356
time: 8.94 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -68.1773148, 54.1091690, -76.5980301, 60.8542404, -129.0315552, 130.7071991
1: -56.3429832, 48.0154800, -63.4378738, 53.9610176, -110.3039932, 111.4533539
2: -74.7554169, 49.2199211, -84.0879135, 55.2139778, -129.9693909, 133.3078308
3: -80.0022659, 41.9368668, -89.7706451, 47.0953178, -127.0975800, 131.7075195
4: -73.3227768, 56.4227982, -82.4272842, 63.4185333, -136.7413025, 138.8500671
5: -65.5744629, 51.1348572, -73.6554718, 57.4330177, -123.0074768, 124.7903290
6: -63.3671684, 60.0410423, -71.0655365, 67.5311279, -130.8982849, 131.1065674
7: -68.9566498, 58.0363235, -77.3692703, 65.0638504, -134.0205078, 135.4055939
8: -81.8438416, 55.6883812, -92.1338272, 62.8546524, -144.6984863, 147.8222046
9: -62.7911453, 61.3247604, -70.5179138, 69.0138245, -131.8049622, 131.8426819

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3517679, upper bound: 197.3527371
time: 8.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3517679, upper bound: 197.4002612
time: 7.61 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -75.8261261, 60.3121338, -57.9427338, 46.0502853, -121.8764114, 118.2548676
1: -62.8633270, 53.4675026, -47.8214684, 40.8501358, -103.7134552, 101.2889709
2: -83.2939453, 54.6399155, -63.5096283, 41.9536591, -125.2476044, 118.1495438
3: -88.8187943, 46.6220093, -68.0391769, 35.6473465, -124.4661255, 114.6611710
4: -81.6678467, 62.8759117, -62.3432198, 48.0855103, -129.7533417, 125.2191238
5: -72.9381943, 56.8863258, -55.8076591, 43.5713387, -116.5095215, 112.6939850
6: -70.3727188, 66.9132614, -54.0077972, 51.0143890, -121.3871078, 120.9210587
7: -76.5561447, 64.4066925, -58.6748962, 49.5177422, -126.0738831, 123.0815811
8: -91.3393250, 62.4127159, -69.5280075, 47.2781334, -138.6174622, 131.9407043
9: -69.7811584, 68.3138657, -53.4291344, 52.0453949, -121.8265533, 121.7429962

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 168

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3524509, upper bound: 197.3522171
time: 7.08 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3980564, upper bound: 197.3974015
time: 6.83 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -78.2179031, 62.1927681, -65.8794403, 52.2954102, -130.5133057, 128.0722046
1: -64.8415833, 55.1365471, -54.3890953, 46.3860855, -111.2276688, 109.5256424
2: -85.9191360, 56.3308945, -72.2206726, 47.5700035, -133.4891357, 128.5515747
3: -91.6161194, 48.0945930, -77.3069305, 40.5152245, -132.1313477, 125.4015198
4: -84.2264862, 64.8117371, -70.8522568, 54.5207787, -138.7472687, 135.6639862
5: -75.2188950, 58.6516495, -63.3780975, 49.4197197, -124.6386108, 122.0297318
6: -72.5468826, 69.0175552, -61.2465439, 57.9995079, -130.5463867, 130.2640991
7: -78.9570618, 66.3858948, -66.6315079, 56.0905304, -135.0475769, 133.0173950
8: -94.2015152, 64.3725281, -79.0618134, 53.7890968, -147.9906158, 143.4343414
9: -71.9602432, 70.4743423, -60.6512718, 59.2058563, -131.1661072, 131.1255951

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 168

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3534659, upper bound: 197.3534748
time: 9.04 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3999677, upper bound: 197.3995213
time: 10.59 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -82.5053101, 65.5929260, -60.2420197, 47.8670578, -130.3723755, 125.8349457
1: -68.5392532, 58.1979141, -49.7753067, 42.4780426, -111.0172958, 107.9732132
2: -90.6620712, 59.4303551, -66.0479279, 43.6061211, -134.2681885, 125.4782867
3: -96.6505051, 50.7477188, -70.7364349, 37.0704155, -133.7209167, 121.4841537
4: -88.8516541, 68.3969727, -64.8158646, 49.9908714, -138.8425140, 133.2128296
5: -79.3276138, 61.8655396, -58.0063400, 45.2878685, -124.6154785, 119.8718796
6: -76.5364380, 72.8436508, -56.1291389, 53.0588417, -129.5952759, 128.9727631
7: -83.3060150, 70.0573196, -61.0022278, 51.4662247, -134.7722015, 131.0595398
8: -99.4258194, 67.9357758, -72.3168259, 49.1797638, -148.6055756, 140.2525787
9: -75.9818802, 74.4452362, -55.5688286, 54.1698914, -130.1517639, 130.0140381

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3540110, upper bound: 197.3540892
time: 8.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3984237, upper bound: 197.3978360
time: 8.10 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -84.8675461, 67.4506226, -68.1773148, 54.1091690, -138.9767151, 135.6278992
1: -70.4933014, 59.8462105, -56.3429832, 48.0154800, -118.5087814, 116.1891937
2: -93.2551422, 61.0984306, -74.7554169, 49.2199211, -142.4750519, 135.8538361
3: -99.4132309, 52.2015228, -80.0022659, 41.9368668, -141.3500977, 132.2037964
4: -91.3773499, 70.3094864, -73.3227768, 56.4227982, -147.8001251, 143.6322632
5: -81.5809326, 63.6083488, -65.5744629, 51.1348572, -132.7157593, 129.1828156
6: -78.6837616, 74.9215012, -63.3671684, 60.0410423, -138.7248077, 138.2886658
7: -85.6768112, 72.0121231, -68.9566498, 58.0363235, -143.7131348, 140.9687653
8: -102.2498474, 69.8722305, -81.8438416, 55.6883812, -157.9382324, 151.7160645
9: -78.1340103, 76.5771179, -62.7911453, 61.3247604, -139.4587555, 139.3682556

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3552974, upper bound: 197.3556151
time: 9.31 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4002612, upper bound: 197.3998998
time: 7.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -75.8261261, 60.3121338, -77.2992630, 61.5009766, -137.3271027, 137.6113739
1: -62.8633270, 53.4675026, -64.3019257, 54.5988350, -117.4621429, 117.7694244
2: -83.2939453, 54.6399155, -84.9845963, 55.7626038, -139.0565186, 139.6244965
3: -88.8187943, 46.6220093, -90.5878448, 47.5759583, -136.3947449, 137.2098236
4: -81.6678467, 62.8759117, -83.2826004, 64.2062607, -145.8740845, 146.1585083
5: -72.9381943, 56.8863258, -74.3595276, 58.0471001, -130.9852753, 131.2458496
6: -70.3727188, 66.9132614, -71.7897568, 68.2973099, -138.6700287, 138.7030182
7: -76.5561447, 64.4066925, -78.1059036, 65.7645874, -142.3207092, 142.5126038
8: -91.3393250, 62.4127159, -93.2075729, 63.7042542, -155.0435791, 155.6202850
9: -69.7811584, 68.3138657, -71.2577591, 69.7733765, -139.5545349, 139.5716248

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 168

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3889231, upper bound: 197.3892909
time: 8.13 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4309055, upper bound: 197.4308090
time: 7.01 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -78.2179031, 62.1927681, -84.8210297, 67.4192200, -145.6371155, 147.0137939
1: -64.8415833, 55.1365471, -70.5221481, 59.8427505, -124.6843338, 125.6586914
2: -85.9191360, 56.3308945, -93.2328949, 61.0774345, -146.9965668, 149.5637817
3: -91.6161194, 48.0945930, -99.3765945, 52.1892471, -143.8053436, 147.4711761
4: -84.2264862, 64.8117371, -91.3349609, 70.2952423, -154.5217133, 156.1466675
5: -75.2188950, 58.6516495, -81.5312271, 63.5877037, -138.8065948, 140.1828766
6: -72.5468826, 69.0175552, -78.6498032, 74.9121475, -147.4590149, 147.6673431
7: -78.9570618, 66.3858948, -85.6495056, 71.9921265, -150.9491882, 152.0354004
8: -94.2015152, 64.3725281, -102.2190781, 69.8545380, -164.0560455, 166.5916138
9: -71.9602432, 70.4743423, -78.0998611, 76.5440521, -148.5042877, 148.5742035

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 168

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3893478, upper bound: 197.3900271
time: 7.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4318249, upper bound: 197.4318312
time: 6.09 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -82.5053101, 65.5929260, -79.8546524, 63.5200233, -146.0253296, 145.4475708
1: -68.5392532, 58.1979141, -66.4755096, 56.4093628, -124.9486084, 124.6734161
2: -90.6620712, 59.4303551, -87.8040314, 57.5958214, -148.2578888, 147.2343903
3: -96.6505051, 50.7477188, -93.5866852, 49.1573524, -145.8078461, 144.3344116
4: -88.8516541, 68.3969727, -86.0329437, 66.3216858, -155.1733398, 154.4299164
5: -79.3276138, 61.8655396, -76.8029633, 59.9532547, -139.2808380, 138.6685028
6: -76.5364380, 72.8436508, -74.1473465, 70.5671692, -147.1035919, 146.9909668
7: -83.3060150, 70.0573196, -80.6907272, 67.9256592, -151.2316589, 150.7480469
8: -99.4258194, 67.9357758, -96.3019333, 65.8203430, -165.2461548, 164.2377014
9: -75.9818802, 74.4452362, -73.6327667, 72.1244125, -148.1062927, 148.0780029

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3899987, upper bound: 197.3905494
time: 6.87 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4315746, upper bound: 197.4315091
time: 6.46 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -84.8675461, 67.4506226, -87.3436508, 69.4137726, -154.2813110, 154.7942352
1: -70.4933014, 59.8462105, -72.6687927, 61.6308365, -132.1241455, 132.5149994
2: -93.2551422, 61.0984306, -96.0182571, 62.8872414, -156.1423798, 157.1166534
3: -99.4132309, 52.2015228, -102.3376160, 53.7498741, -153.1631012, 154.5391235
4: -91.3773499, 70.3094864, -94.0491791, 72.3854446, -163.7627716, 164.3586731
5: -81.5809326, 63.6083488, -83.9450302, 65.4696503, -147.0505829, 147.5533752
6: -78.6837616, 74.9215012, -80.9792633, 77.1534271, -155.8371887, 155.9007568
7: -85.6768112, 72.0121231, -88.2022552, 74.1273880, -159.8041992, 160.2143250
8: -102.2498474, 69.8722305, -105.2730026, 71.9457550, -174.1955872, 175.1452332
9: -78.1340103, 76.5771179, -80.4464798, 78.8647079, -156.9986877, 157.0235901

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3903241, upper bound: 197.3911440
time: 9.04 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4325254, upper bound: 197.4325262
time: 8.33 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 18.86 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3435790, upper bound: 197.3448669
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3914039, upper bound: 197.3917330
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3435790, upper bound: 197.3509764
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3914039, upper bound: 197.3917330
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3445014, upper bound: 197.3455256
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3935833, upper bound: 197.3936362
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3445014, upper bound: 197.3455256
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3935833, upper bound: 197.3936362
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3446313, upper bound: 197.3458486
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3916919, upper bound: 197.3919603
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3446313, upper bound: 197.3517804
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3916919, upper bound: 197.3984237
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3429292, upper bound: 197.3423829
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3552385, upper bound: 197.3549356
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3517679, upper bound: 197.3527371
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3517679, upper bound: 197.4002612
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3524509, upper bound: 197.3522171
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3980564, upper bound: 197.3974015
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3534659, upper bound: 197.3534748
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3999677, upper bound: 197.3995213
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3540110, upper bound: 197.3540892
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3984237, upper bound: 197.3978360
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3552974, upper bound: 197.3556151
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.4002612, upper bound: 197.3998998
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3889231, upper bound: 197.3892909
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.4309055, upper bound: 197.4308090
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3893478, upper bound: 197.3900271
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.4318249, upper bound: 197.4318312
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3899987, upper bound: 197.3905494
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.4315746, upper bound: 197.4315091
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.3903241, upper bound: 197.3911440
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.86
Output dim: 4, lower bound: -197.4325254, upper bound: 197.4325262

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -47.4679146, 37.6418762, -51.1771278, 40.6312370, -88.0991516, 88.8190002
1: -38.8449783, 33.3431625, -41.8857231, 35.9514542, -74.7964325, 75.2288666
2: -51.9211960, 34.4692459, -55.9766922, 37.0592918, -88.9804840, 90.4459229
3: -55.8201180, 29.1703510, -60.0769081, 31.4306316, -87.2507477, 89.2472458
4: -51.0849304, 39.2998428, -55.0345345, 42.3612747, -93.4462051, 94.3343811
5: -45.7621841, 35.7236023, -49.3151779, 38.4544220, -84.2166061, 85.0387802
6: -44.3253365, 41.6139870, -47.7023163, 44.9141312, -89.2394714, 89.3162994
7: -48.1248817, 40.6121941, -51.7996445, 43.6828079, -91.8076935, 92.4118271
8: -56.5187950, 38.2916603, -61.1576614, 41.5104675, -98.0292587, 99.4493256
9: -43.7176476, 42.3769341, -47.0778770, 45.7437401, -89.4613800, 89.4547958

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3306206, upper bound: 197.3315094
time: 6.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3431080, upper bound: 197.3442736
time: 7.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -55.3833160, 44.0194168, -56.8708534, 45.1799011, -100.5632172, 100.8902740
1: -45.6325645, 39.0242767, -46.7256660, 40.0073929, -85.6399536, 85.7499390
2: -60.6905785, 40.1326218, -62.2637253, 41.1164856, -101.8070526, 102.3963470
3: -65.0504074, 34.0663147, -66.7398453, 34.9410629, -99.9914627, 100.8061600
4: -59.5827065, 45.9672623, -61.1706085, 47.0911636, -106.6738739, 107.1378632
5: -53.3589630, 41.6650772, -54.7701302, 42.7085228, -96.0674667, 96.4352112
6: -51.6435699, 48.7329788, -52.9590340, 49.9874458, -101.6310120, 101.6920090
7: -56.1016235, 47.3591499, -57.5324364, 48.4921188, -104.5937424, 104.8915710
8: -66.4026184, 45.1274033, -68.1527634, 46.3337097, -112.7363205, 113.2801590
9: -51.0727882, 49.7116470, -52.3251114, 50.9505348, -102.0233231, 102.0367508

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3439151, upper bound: 197.3430297
time: 6.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3439151, upper bound: 197.3917330
time: 7.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -47.4679146, 37.6418762, -69.5849533, 55.3434868, -102.8114014, 107.2268295
1: -38.8449783, 33.3431625, -57.5094337, 49.0119171, -87.8568954, 90.8525848
2: -51.9211960, 34.4692459, -76.4041748, 50.2001266, -102.1213226, 110.8734131
3: -55.8201180, 29.1703510, -81.5516968, 42.7532845, -98.5733948, 110.7220459
4: -51.0849304, 39.2998428, -74.9236908, 57.6881180, -108.7730484, 114.2235336
5: -45.7621841, 35.7236023, -66.9579086, 52.2439461, -98.0061188, 102.6815109
6: -44.3253365, 41.6139870, -64.6052551, 61.3337822, -105.6591034, 106.2192383
7: -48.1248817, 40.6121941, -70.2841644, 59.1597023, -107.2845840, 110.8963623
8: -56.5187950, 38.2916603, -83.7072678, 57.1163826, -113.6351776, 121.9989243
9: -43.7176476, 42.3769341, -64.0404968, 62.6071358, -106.3247757, 106.4174347

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3319312, upper bound: 197.3332771
time: 8.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3330548, upper bound: 197.3345685
time: 8.12 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -55.3833160, 44.0194168, -75.8119965, 60.2978096, -115.6811218, 119.8314133
1: -45.6325645, 39.0242767, -62.8509064, 53.4563637, -99.0889282, 101.8751755
2: -60.6905785, 40.1326218, -83.2740860, 54.6312675, -115.3218460, 123.4067078
3: -65.0504074, 34.0663147, -88.8035431, 46.6117477, -111.6621552, 122.8698578
4: -59.5827065, 45.9672623, -81.6525497, 62.8625793, -122.4452744, 127.6198120
5: -53.3589630, 41.6650772, -72.9256134, 56.8753815, -110.2343140, 114.5906906
6: -51.6435699, 48.7329788, -70.3589935, 66.8992462, -118.5428085, 119.0919724
7: -56.1016235, 47.3591499, -76.5403214, 64.3945618, -120.4961853, 123.8994675
8: -66.4026184, 45.1274033, -91.3225021, 62.3986130, -128.8012238, 136.4499054
9: -51.0727882, 49.7116470, -69.7656937, 68.2994537, -119.3722382, 119.4773407

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 168

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3522171, upper bound: 197.3524509
time: 6.52 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3522171, upper bound: 197.3980564
time: 7.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -54.7215843, 43.3499985, -53.5525131, 42.4985390, -97.2201080, 96.9024963
1: -44.7856941, 38.3751106, -43.8243408, 37.6009789, -82.3866577, 82.1994476
2: -59.8806915, 39.5832481, -58.5819130, 38.7380905, -98.6187820, 98.1651611
3: -64.3329544, 33.5935249, -62.8623543, 32.8848343, -97.2177887, 96.4558792
4: -58.8240967, 45.1662483, -57.5734825, 44.2801704, -103.1042633, 102.7397308
5: -52.7031136, 41.0701408, -51.5768814, 40.2142334, -92.9173431, 92.6470184
6: -50.9307022, 47.9796181, -49.8584633, 46.9951553, -97.9258575, 97.8380814
7: -55.3959122, 46.6520004, -54.1852798, 45.6539421, -101.0498505, 100.8372803
8: -65.2225189, 44.1653137, -64.0118866, 43.4517593, -108.6742783, 108.1772003
9: -50.3221130, 48.8782005, -49.2435493, 47.8847427, -98.2068558, 98.1217346

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3309015, upper bound: 197.3317689
time: 8.06 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3440366, upper bound: 197.3450530
time: 6.74 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 16.28 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 16.28
Output dim: 4, lower bound: -197.3306206, upper bound: 197.3315094
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 16.28
Output dim: 4, lower bound: -197.3431080, upper bound: 197.3442736
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 16.28
Output dim: 4, lower bound: -197.3439151, upper bound: 197.3430297
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 16.28
Output dim: 4, lower bound: -197.3439151, upper bound: 197.3917330
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 16.28
Output dim: 4, lower bound: -197.3319312, upper bound: 197.3332771
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 16.28
Output dim: 4, lower bound: -197.3330548, upper bound: 197.3345685
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 16.28
Output dim: 4, lower bound: -197.3522171, upper bound: 197.3524509
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 16.28
Output dim: 4, lower bound: -197.3522171, upper bound: 197.3980564
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 16.28
Output dim: 4, lower bound: -197.3309015, upper bound: 197.3317689
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 16.28
Output dim: 4, lower bound: -197.3440366, upper bound: 197.3450530
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3935833, upper bound: 197.3936362
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3445014, upper bound: 197.3455256
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3935833, upper bound: 197.3936362
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3446313, upper bound: 197.3458486
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3916919, upper bound: 197.3919603
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3446313, upper bound: 197.3517804
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3916919, upper bound: 197.3984237
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3429292, upper bound: 197.3423829
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3552385, upper bound: 197.3549356
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3517679, upper bound: 197.3527371
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3517679, upper bound: 197.4002612
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3524509, upper bound: 197.3522171
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3980564, upper bound: 197.3974015
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3534659, upper bound: 197.3534748
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3999677, upper bound: 197.3995213
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3540110, upper bound: 197.3540892
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3984237, upper bound: 197.3978360
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3552974, upper bound: 197.3556151
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.4002612, upper bound: 197.3998998
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3889231, upper bound: 197.3892909
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.4309055, upper bound: 197.4308090
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3893478, upper bound: 197.3900271
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.4318249, upper bound: 197.4318312
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3899987, upper bound: 197.3905494
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.4315746, upper bound: 197.4315091
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.3903241, upper bound: 197.3911440
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 4, lower bound: -197.4325254, upper bound: 197.4325262
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=198.953369140625
rel_dist={4: [-197.44083159618555, 197.44083163160866]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4055316, upper bound: 197.4057404
time: 8.77 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4387864, upper bound: 197.4387864
time: 7.15 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.06 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.06
Output dim: 4, lower bound: -197.4055316, upper bound: 197.4057404
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.06
Output dim: 4, lower bound: -197.4387864, upper bound: 197.4387864

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -79.9444504, 63.3750114, -84.9462128, 67.3801575, -147.3246002, 148.3212128
1: -66.2200775, 56.2719460, -70.4969406, 59.8315163, -126.0515747, 126.7688751
2: -87.7009277, 57.5595970, -93.2516708, 61.1313972, -148.8322906, 150.8112640
3: -93.7524338, 49.1665268, -99.5563278, 52.2514153, -146.0038452, 148.7227936
4: -85.9368134, 66.0450745, -91.3321457, 70.2281952, -156.1650085, 157.3772278
5: -76.8184204, 59.8392181, -81.6064453, 63.5883827, -140.4067993, 141.4456635
6: -74.1077042, 70.4479294, -78.7007751, 74.9175797, -149.0252838, 149.1487122
7: -80.7441788, 67.8362045, -85.7653732, 72.0295029, -152.7736664, 153.6015778
8: -96.0069962, 65.3933868, -102.1313629, 69.6759109, -165.6829071, 167.5247498
9: -73.5597076, 71.9806671, -78.1796570, 76.5840530, -150.1437378, 150.1603241

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3996141, upper bound: 197.3997959
time: 8.06 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3999306, upper bound: 197.4000410
time: 10.60 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -99.1355057, 78.7032318, -100.8557587, 80.0711365, -179.2065887, 179.5589752
1: -82.5673141, 69.9038849, -84.0246735, 71.1205444, -153.6878357, 153.9285278
2: -108.9954758, 71.2377625, -110.8976669, 72.4636307, -181.4591064, 182.1354370
3: -116.1185074, 61.0004578, -118.1280594, 62.0610466, -178.1795349, 179.1285095
4: -106.6865845, 82.0365067, -108.5423965, 83.4567032, -190.1432800, 190.5788574
5: -95.2138977, 74.1955032, -96.8637314, 75.4787064, -170.6925812, 171.0592194
6: -91.7468567, 87.5859756, -93.3255234, 89.1156998, -180.8625488, 180.9114990
7: -100.0214844, 83.9518967, -101.7495270, 85.3969498, -185.4184113, 185.7014160
8: -119.4582748, 81.6738663, -121.5436096, 83.1101685, -202.5684357, 203.2174683
9: -91.2429123, 89.5477371, -92.8219528, 91.1091843, -182.3520966, 182.3696899

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4348246, upper bound: 197.4348242
time: 8.36 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4352634, upper bound: 197.4352634
time: 4.95 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 14.83 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.83
Output dim: 4, lower bound: -197.3996141, upper bound: 197.3997959
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.83
Output dim: 4, lower bound: -197.3999306, upper bound: 197.4000410
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.83
Output dim: 4, lower bound: -197.4348246, upper bound: 197.4348242
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.83
Output dim: 4, lower bound: -197.4352634, upper bound: 197.4352634

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -67.2402649, 53.3487434, -67.1191635, 53.3082466, -120.5485077, 120.4679031
1: -55.4211235, 47.2830925, -55.3513031, 47.2238693, -102.6449890, 102.6343918
2: -73.6745300, 48.4829865, -73.5734482, 48.4032478, -122.0777664, 122.0564346
3: -78.8630600, 41.3184738, -78.6695786, 41.2449303, -120.1079865, 119.9880447
4: -72.2884827, 55.5650864, -72.1825943, 55.5325699, -127.8210526, 127.7476807
5: -64.6803589, 50.3857079, -64.5708084, 50.3284683, -115.0088272, 114.9565125
6: -62.4387283, 59.1554337, -62.3243446, 59.0764198, -121.5151367, 121.4797745
7: -67.9349136, 57.1514359, -67.7984161, 57.0344009, -124.9693146, 124.9498367
8: -80.6403885, 54.8706245, -80.5770721, 54.9237480, -135.5641327, 135.4476929
9: -61.8224030, 60.3595047, -61.7217712, 60.2997513, -122.1221542, 122.0812759

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3949154, upper bound: 197.3952851
time: 8.09 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3975502, upper bound: 197.3977261
time: 9.14 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -71.1809158, 56.4608078, -73.3372498, 58.2178116, -129.3986969, 129.7980652
1: -58.7729607, 50.0766106, -60.6299095, 51.6233101, -110.3962708, 110.7065048
2: -78.0240173, 51.3141365, -80.4277649, 52.8623581, -130.8863525, 131.7418976
3: -83.4865112, 43.7555237, -85.9569092, 45.0810394, -128.5675354, 129.7124023
4: -76.5263062, 58.8284035, -78.8664627, 60.6659698, -137.1922150, 137.6948547
5: -68.4477386, 53.3282204, -70.5162659, 54.9627953, -123.4105301, 123.8444824
6: -66.0763931, 62.6583405, -68.0608749, 64.5958099, -130.6722107, 130.7192078
7: -71.9246368, 60.4916611, -74.0797958, 62.2975540, -134.2221832, 134.5714569
8: -85.4175262, 58.1258621, -88.1041794, 60.0481567, -145.4656830, 146.2300262
9: -65.4933243, 64.0007706, -67.4940643, 66.0107422, -131.5040588, 131.4948120

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3953335, upper bound: 197.3956402
time: 10.03 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3978171, upper bound: 197.3979727
time: 8.21 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -86.0924759, 68.4023895, -82.7185287, 65.7483444, -151.8407898, 151.1208954
1: -71.4793472, 60.6769753, -68.6167068, 58.2968292, -129.7761230, 129.2936707
2: -94.5912018, 61.9252243, -90.8750305, 59.5190048, -154.1101990, 152.8002625
3: -100.8238678, 52.9394341, -96.8682861, 50.8593864, -151.6832428, 149.8076935
4: -92.6716995, 71.2711334, -89.0582428, 68.4981766, -161.1698456, 160.3293762
5: -82.7467575, 64.4883041, -79.5255051, 61.9861832, -144.7329407, 144.0138092
6: -79.7604675, 75.9911270, -76.6570358, 72.9994354, -152.7599030, 152.6481628
7: -86.8633270, 72.9762192, -83.4624176, 70.1367035, -157.0000305, 156.4386292
8: -103.6907883, 70.8687744, -99.6238937, 68.0994186, -171.7901917, 170.4926758
9: -79.1918182, 77.6219559, -76.0776749, 74.5474091, -153.7392273, 153.6996307

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4314011, upper bound: 197.4314320
time: 6.50 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4324849, upper bound: 197.4324846
time: 6.43 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -90.4619827, 71.8600998, -89.3513565, 70.9938507, -161.4558411, 161.2114563
1: -75.2005768, 63.7770386, -74.2526627, 62.9940453, -138.1946259, 138.0296936
2: -99.4175339, 65.0636444, -98.1934204, 64.2751694, -163.6927032, 163.2570648
3: -105.9538116, 55.6439590, -104.6464539, 54.9557533, -160.9095459, 160.2904053
4: -97.3752518, 74.8936539, -96.1919250, 73.9820557, -171.3572998, 171.0855713
5: -86.9273758, 67.7514191, -85.8719635, 66.9313736, -153.8587494, 153.6233673
6: -83.7953568, 79.8762589, -82.7792053, 78.8892670, -162.6846008, 162.6554565
7: -91.2887955, 76.6789017, -90.1667328, 75.7500687, -167.0388641, 166.8456116
8: -108.9861526, 74.4924622, -107.6537170, 73.5835800, -182.5697327, 182.1461792
9: -83.2611694, 81.6512070, -82.2360764, 80.6358109, -163.8969727, 163.8872681

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4320492, upper bound: 197.4320705
time: 8.59 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4331099, upper bound: 197.4331099
time: 7.05 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 17.14 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.14
Output dim: 4, lower bound: -197.3949154, upper bound: 197.3952851
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.14
Output dim: 4, lower bound: -197.3975502, upper bound: 197.3977261
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.14
Output dim: 4, lower bound: -197.3953335, upper bound: 197.3956402
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.14
Output dim: 4, lower bound: -197.3978171, upper bound: 197.3979727
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.14
Output dim: 4, lower bound: -197.4314011, upper bound: 197.4314320
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.14
Output dim: 4, lower bound: -197.4324849, upper bound: 197.4324846
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.14
Output dim: 4, lower bound: -197.4320492, upper bound: 197.4320705
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.14
Output dim: 4, lower bound: -197.4331099, upper bound: 197.4331099

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -52.3659668, 41.6514740, -57.2698479, 45.5663605, -97.9323273, 98.9213257
1: -43.0816422, 36.9012604, -47.1720657, 40.3476067, -83.4292297, 84.0733109
2: -57.3580666, 37.9683647, -62.7656784, 41.4437485, -98.8018188, 100.7340393
3: -61.4827728, 32.2023430, -67.1713028, 35.1950722, -96.6778259, 99.3736420
4: -56.3570175, 43.4957428, -61.6391525, 47.5335808, -103.8905945, 105.1348801
5: -50.4801865, 39.4135971, -55.1693459, 43.0618057, -93.5419922, 94.5829468
6: -48.8843536, 46.0668106, -53.3656311, 50.4035873, -99.2879181, 99.4324265
7: -53.0445290, 44.8121719, -57.9433060, 48.8796043, -101.9241257, 102.7554626
8: -62.7904129, 42.6828194, -68.7677994, 46.8245773, -109.6149826, 111.4506073
9: -48.2731628, 46.9416237, -52.7433739, 51.4008179, -99.6739731, 99.6849976

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3401853, upper bound: 197.3403417
time: 9.02 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3931632, upper bound: 197.3934385
time: 7.70 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -60.1127167, 47.7457352, -61.8765717, 49.1879883, -109.3007050, 109.6222916
1: -49.4893494, 42.3089256, -50.9864235, 43.5648193, -93.0541611, 93.2953491
2: -65.8558350, 43.4495888, -67.8238831, 44.7029762, -110.5588074, 111.2734680
3: -70.5514069, 36.9512787, -72.5554352, 38.0323906, -108.5837784, 109.5067139
4: -64.6583176, 49.7657127, -66.5717468, 51.2677193, -115.9260406, 116.3374557
5: -57.8694115, 45.1294556, -59.5626068, 46.4624252, -104.3318329, 104.6920624
6: -55.9510765, 52.8743477, -57.5532417, 54.4576416, -110.4087143, 110.4275894
7: -60.8194962, 51.2400818, -62.5642662, 52.6896057, -113.5090942, 113.8043518
8: -72.0884628, 49.0138512, -74.2906113, 50.6138725, -122.7023239, 123.3044586
9: -55.3238258, 53.9282608, -56.9399376, 55.5712738, -110.8950882, 110.8681946

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3412250, upper bound: 197.3413302
time: 10.41 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3955439, upper bound: 197.3956275
time: 9.80 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -56.1618690, 44.6531334, -63.4793243, 50.4689903, -106.6308517, 108.1324615
1: -46.3096695, 39.5961685, -52.4432335, 44.7394257, -91.0490799, 92.0393982
2: -61.5475197, 40.6989250, -69.6096802, 45.8975868, -107.4450912, 110.3085938
3: -65.9553223, 34.5512047, -74.4468384, 39.0277748, -104.9830856, 108.9980316
4: -60.4357033, 46.6358299, -68.3122864, 52.6592865, -113.0949860, 114.9481049
5: -54.1120377, 42.2563057, -61.1061020, 47.6891289, -101.8011627, 103.3624115
6: -52.3906021, 49.4355240, -59.0926094, 55.9160614, -108.3066635, 108.5281296
7: -56.8957748, 48.0440216, -64.2130814, 54.1351814, -111.0309525, 112.2570877
8: -67.3899155, 45.8056908, -76.2879715, 51.9445686, -119.3344727, 122.0936508
9: -51.8147964, 50.4557686, -58.5080032, 57.1078491, -108.9226379, 108.9637604

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3416988, upper bound: 197.3418332
time: 8.48 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3935662, upper bound: 197.3937971
time: 11.42 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -64.0246048, 50.8332481, -68.0429077, 54.0553055, -118.0799103, 118.8761597
1: -52.8128433, 45.0795860, -56.2219582, 47.9277992, -100.7406464, 101.3015366
2: -70.1699524, 46.2603912, -74.6185532, 49.1255875, -119.2955399, 120.8789444
3: -75.1388092, 39.3728104, -79.7804184, 41.8387032, -116.9775085, 119.1532135
4: -68.8616486, 53.0051117, -73.1977844, 56.3563499, -125.2179871, 126.2028885
5: -61.6083679, 48.0500984, -65.4578705, 51.0579643, -112.6663361, 113.5079651
6: -59.5619583, 56.3495293, -63.2418518, 59.9296799, -119.4916382, 119.5913849
7: -64.7787552, 54.5571785, -68.7925949, 57.9102974, -122.6890564, 123.3497772
8: -76.8249435, 52.2440987, -81.7514648, 55.6942673, -132.5192108, 133.9955597
9: -58.9695244, 57.5426559, -62.6671143, 61.2335320, -120.2030563, 120.2097626

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3431209, upper bound: 197.3432336
time: 9.19 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3958005, upper bound: 197.3958594
time: 10.10 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -71.3798065, 56.8282013, -72.9905777, 58.0982780, -129.4780731, 129.8187714
1: -59.2683640, 50.4094620, -60.5390587, 51.5040092, -110.7723694, 110.9485168
2: -78.4476852, 51.5364151, -80.1961441, 52.6496582, -131.0973511, 131.7325592
3: -83.6504211, 43.9174042, -85.5077438, 44.8830833, -128.5335083, 129.4251099
4: -76.9215775, 59.3209076, -78.6489182, 60.5947914, -137.5163727, 137.9698181
5: -68.7042236, 53.6421700, -70.2391357, 54.8078728, -123.5120926, 123.8812943
6: -66.3507462, 63.0348625, -67.8056335, 64.4307175, -130.7814636, 130.8404999
7: -72.1359329, 60.7858849, -73.7224350, 62.0785179, -134.2144012, 134.5083160
8: -86.0507736, 58.7978554, -87.9683990, 60.1058655, -146.1566315, 146.7662506
9: -65.7876205, 64.3587799, -67.2061691, 65.7628174, -131.5504150, 131.5649414

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3835343, upper bound: 197.3834914
time: 8.73 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4305040, upper bound: 197.4305572
time: 7.65 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -78.9237366, 62.7619705, -77.3847656, 61.5525284, -140.4762573, 140.1467285
1: -65.5072098, 55.6690865, -64.1737061, 54.5714569, -120.0786591, 119.8427887
2: -86.7195282, 56.8667984, -85.0189667, 55.7557831, -142.4753113, 141.8857727
3: -92.4643021, 48.5448303, -90.6483917, 47.5899277, -140.0542145, 139.1931915
4: -84.9991302, 65.4273987, -83.3497314, 64.1513138, -149.1504211, 148.7771301
5: -75.8956680, 59.1989021, -74.4286575, 58.0512123, -133.9468689, 133.6275482
6: -73.2310791, 69.6684494, -71.7997131, 68.2959976, -141.5270691, 141.4681396
7: -79.7009659, 67.0312042, -78.1336365, 65.7140274, -145.4149933, 145.1648254
8: -95.0889359, 64.9654312, -93.2247238, 63.7077103, -158.7966461, 158.1901550
9: -72.6490402, 71.1501846, -71.2097397, 69.7328415, -142.3818817, 142.3599243

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 168

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3841790, upper bound: 197.3840815
time: 9.75 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4316499, upper bound: 197.4316582
time: 12.10 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -75.7740555, 60.3015480, -79.6376953, 63.3522301, -139.1262665, 139.9392395
1: -63.0085449, 53.5266266, -66.1865387, 56.2117958, -119.2203369, 119.7131653
2: -83.2988663, 54.6931953, -87.5283585, 57.4174232, -140.7162628, 142.2215424
3: -88.8054199, 46.6363640, -93.3007431, 48.9885902, -137.7940063, 139.9371033
4: -81.6518326, 62.9618111, -85.7984848, 66.0888290, -147.7406616, 148.7602997
5: -72.9056549, 56.9223404, -76.5972366, 59.7627182, -132.6683655, 133.5195770
6: -70.4068680, 66.9409943, -73.9401245, 70.3328094, -140.7396393, 140.8810883
7: -76.5833054, 64.5061951, -80.4397964, 67.7020874, -144.2853699, 144.9459839
8: -91.3769073, 62.4405174, -96.0165634, 65.6016006, -156.9785004, 158.4570770
9: -69.8783798, 68.4100113, -73.3778687, 71.8649292, -141.7433167, 141.7878723

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3846263, upper bound: 197.3845276
time: 10.01 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4313008, upper bound: 197.4313171
time: 8.36 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -83.2432861, 66.1784821, -83.9658508, 66.7554092, -149.9986572, 150.1443329
1: -69.1858139, 58.7345772, -69.7666702, 59.2324905, -128.4182892, 128.5012512
2: -91.4905090, 59.9688072, -92.2792130, 60.4747620, -151.9652710, 152.2480164
3: -97.5323715, 51.2174225, -98.3633499, 51.6537132, -149.1860352, 149.5807495
4: -89.6466599, 69.0093765, -90.4256592, 69.5927963, -159.2394562, 159.4350281
5: -80.0283356, 62.4236221, -80.7249298, 62.9568787, -142.9852142, 143.1485596
6: -77.2210388, 73.5090485, -77.8742447, 74.1392670, -151.3602753, 151.3832397
7: -84.0744934, 70.6905136, -84.7841110, 71.2829208, -155.3574219, 155.4746246
8: -100.3233261, 68.5494995, -101.1910629, 69.1506119, -169.4739227, 169.7405548
9: -76.6736603, 75.1311111, -77.3215256, 75.7716980, -152.4453430, 152.4526215

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3851721, upper bound: 197.3849824
time: 9.52 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4324535, upper bound: 197.4324535
time: 9.94 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 20.96 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.96
Output dim: 4, lower bound: -197.3401853, upper bound: 197.3403417
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.96
Output dim: 4, lower bound: -197.3931632, upper bound: 197.3934385
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.96
Output dim: 4, lower bound: -197.3412250, upper bound: 197.3413302
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.96
Output dim: 4, lower bound: -197.3955439, upper bound: 197.3956275
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.96
Output dim: 4, lower bound: -197.3416988, upper bound: 197.3418332
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.96
Output dim: 4, lower bound: -197.3935662, upper bound: 197.3937971
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.96
Output dim: 4, lower bound: -197.3431209, upper bound: 197.3432336
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.96
Output dim: 4, lower bound: -197.3958005, upper bound: 197.3958594
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.96
Output dim: 4, lower bound: -197.3835343, upper bound: 197.3834914
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.96
Output dim: 4, lower bound: -197.4305040, upper bound: 197.4305572
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.96
Output dim: 4, lower bound: -197.3841790, upper bound: 197.3840815
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.96
Output dim: 4, lower bound: -197.4316499, upper bound: 197.4316582
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.96
Output dim: 4, lower bound: -197.3846263, upper bound: 197.3845276
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.96
Output dim: 4, lower bound: -197.4313008, upper bound: 197.4313171
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.96
Output dim: 4, lower bound: -197.3851721, upper bound: 197.3849824
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.96
Output dim: 4, lower bound: -197.4324535, upper bound: 197.4324535

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -42.2745361, 33.5802269, -46.5507698, 36.9627304, -79.2372437, 80.1309814
1: -34.5536880, 29.7507610, -38.0090294, 32.6784134, -67.2321014, 67.7597885
2: -46.1944847, 30.7923851, -50.9030991, 33.7970085, -79.9914932, 81.6954803
3: -49.6552238, 25.9993896, -54.6789856, 28.5754528, -78.2306747, 80.6783524
4: -45.5196648, 35.1024742, -50.1024323, 38.5275192, -84.0471802, 85.2048874
5: -40.7760620, 31.8849735, -44.8961029, 35.0234032, -75.7994537, 76.7810745
6: -39.5681229, 37.0635605, -43.4722939, 40.7923470, -80.3604736, 80.5358429
7: -42.9001999, 36.2486343, -47.1492996, 39.7794876, -82.6796875, 83.3979340
8: -50.3607368, 34.1762466, -55.4659424, 37.6008110, -87.9615250, 89.6421890
9: -38.9666901, 37.7469254, -42.8139992, 41.5085487, -80.4752350, 80.5609207

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2694769, upper bound: 197.2693909
time: 7.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2659924, upper bound: 197.2660871
time: 10.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -50.9069633, 40.4896889, -54.6247368, 43.4664268, -94.3733902, 95.1144104
1: -41.8379135, 35.8610954, -44.9161148, 38.4679832, -80.3058929, 80.7772064
2: -55.7485886, 36.9322510, -59.8520050, 39.5650635, -95.3136444, 96.7842484
3: -59.7745056, 31.3005161, -64.0922852, 33.5563469, -93.3308411, 95.3927994
4: -54.7853241, 42.2856255, -58.7826920, 45.3414688, -100.1267929, 101.0683136
5: -49.0810623, 38.3258400, -52.6363945, 41.0938683, -90.1749115, 90.9622192
6: -47.5366325, 44.7685738, -50.9228935, 48.0470200, -95.5836487, 95.6914597
7: -51.5775299, 43.5802116, -55.2886772, 46.6557007, -98.2332306, 98.8688889
8: -61.0070076, 41.4550209, -65.5408936, 44.5946312, -105.6016388, 106.9959106
9: -46.9296417, 45.6088829, -50.3147163, 48.9861412, -95.9157867, 95.9235916

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3798935, upper bound: 197.3796932
time: 8.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3931427, upper bound: 197.3934385
time: 9.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -49.4169502, 39.2060432, -50.6784821, 40.2100410, -89.6269836, 89.8845215
1: -40.4085388, 34.7070084, -41.3794975, 35.5429039, -75.9514389, 76.0865021
2: -54.0385437, 35.8320885, -55.4348297, 36.7056236, -90.7441559, 91.2668915
3: -58.0462189, 30.3646278, -59.5251541, 31.1095505, -89.1557693, 89.8897705
4: -53.1469536, 40.8821831, -54.5002937, 41.8622971, -95.0092468, 95.3824692
5: -47.6115036, 37.1511688, -48.8399544, 38.0755157, -85.6870193, 85.9911041
6: -46.0797691, 43.3329086, -47.2132492, 44.4115219, -90.4912872, 90.5461578
7: -50.0666618, 42.2031937, -51.2922325, 43.2130356, -93.2796936, 93.4954071
8: -58.9398766, 39.9610062, -60.4143066, 40.9450073, -99.8848877, 100.3753128
9: -45.4719009, 44.1534233, -46.5767860, 45.2173157, -90.6891937, 90.7302094

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3265362, upper bound: 197.3265268
time: 8.98 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3410271, upper bound: 197.3410494
time: 8.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -58.5326881, 46.4905815, -59.2211380, 47.0801239, -105.6128082, 105.7117157
1: -48.1422539, 41.1868744, -48.7188377, 41.6757584, -89.8180084, 89.9057083
2: -64.1145172, 42.3263741, -64.8987350, 42.8158569, -106.9303741, 107.2250977
3: -68.7134857, 35.9738884, -69.4664154, 36.3885536, -105.1020355, 105.4403076
4: -62.9533539, 48.4557114, -63.7053146, 49.0672302, -112.0205688, 112.1610107
5: -56.3565331, 43.9546738, -57.0211601, 44.4878044, -100.8443298, 100.9758148
6: -54.4928856, 51.4652100, -55.1020317, 52.0893326, -106.5822144, 106.5672455
7: -59.2338867, 49.9123840, -59.8998528, 50.4583359, -109.6922226, 109.8122406
8: -70.1603775, 47.6808090, -71.0518646, 48.3741570, -118.5345306, 118.7326736
9: -53.8727531, 52.4857368, -54.5009460, 53.1490097, -107.0217590, 106.9866791

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3790868, upper bound: 197.3789717
time: 9.14 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3955439, upper bound: 197.3956275
time: 8.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -45.8574219, 36.4238129, -52.1653862, 41.4166832, -87.2741089, 88.5891876
1: -37.5662804, 32.2745285, -42.7168846, 36.6273041, -74.1935883, 74.9914093
2: -50.1612968, 33.3640709, -57.1026878, 37.8152275, -87.9765244, 90.4667587
3: -53.8897247, 28.2014561, -61.3024139, 32.0166664, -85.9063873, 89.5038681
4: -49.3523636, 38.0784264, -56.1023979, 43.1775398, -92.5298920, 94.1808243
5: -44.2158813, 34.5722885, -50.2894821, 39.2250977, -83.4409637, 84.8617706
6: -42.8710938, 40.2411423, -48.6392555, 45.7632332, -88.6343231, 88.8803787
7: -46.5354156, 39.3192444, -52.8275719, 44.5761414, -91.1115570, 92.1468201
8: -54.7144737, 37.1087418, -62.2783775, 42.1934357, -96.9079132, 99.3871155
9: -42.3168793, 41.0565872, -48.0447617, 46.6648216, -88.9816971, 89.1013489

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2741080, upper bound: 197.2740022
time: 7.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2706947, upper bound: 197.2707914
time: 8.18 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -54.6381302, 43.4429054, -60.7781067, 48.3251038, -102.9632339, 104.2210083
1: -45.0049515, 38.5073051, -50.1340828, 42.8167381, -87.8216858, 88.6413727
2: -59.8677254, 39.6140976, -66.6326370, 43.9791489, -103.8468628, 106.2467346
3: -64.1746597, 33.6097832, -71.3037643, 37.3562965, -101.5309601, 104.9135437
4: -58.7924156, 45.3755341, -65.3956680, 50.4218483, -109.2142639, 110.7712021
5: -52.6540337, 41.1203384, -58.5207443, 45.6808815, -98.3349075, 99.6410675
6: -50.9823685, 48.0768471, -56.5990562, 53.5064201, -104.4887848, 104.6758957
7: -55.3630867, 46.7569923, -61.5028648, 51.8657608, -107.2288513, 108.2598495
8: -65.5283737, 44.5256615, -72.9929504, 49.6671104, -115.1954727, 117.5186157
9: -50.4105644, 49.0658379, -56.0261078, 54.6434288, -105.0539932, 105.0919342

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3812977, upper bound: 197.3811039
time: 9.43 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3935476, upper bound: 197.3937971
time: 8.65 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -53.0245514, 42.0580406, -56.3442841, 44.7030525, -97.7275925, 98.4023285
1: -43.4348564, 37.2459602, -46.1264648, 39.5267258, -82.9615784, 83.3724213
2: -58.0242577, 38.4167061, -61.6910820, 40.7653389, -98.7895966, 100.1077881
3: -62.3041992, 32.5767403, -66.2010193, 34.5776787, -96.8818817, 98.7777481
4: -57.0023346, 43.8716469, -60.5610962, 46.5488167, -103.5511475, 104.4327393
5: -51.0688667, 39.8534698, -54.2735977, 42.3171005, -93.3859711, 94.1270676
6: -49.4006882, 46.5276413, -52.4322243, 49.4258194, -98.8264923, 98.9598694
7: -53.7201462, 45.2857628, -57.0206604, 48.0462112, -101.7663574, 102.3064117
8: -63.3112373, 42.9149895, -67.2973633, 45.6001663, -108.9114075, 110.2123489
9: -48.8406487, 47.4802971, -51.8540535, 50.4246597, -99.2653046, 99.3343506

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3289167, upper bound: 197.3288545
time: 8.47 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3430761, upper bound: 197.3430823
time: 9.14 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -62.4339218, 49.5706673, -65.3610840, 51.9270134, -114.3609314, 114.9317474
1: -51.4540176, 43.9478683, -53.9301949, 46.0186539, -97.4726715, 97.8780365
2: -68.4170380, 45.1306572, -71.6633224, 47.2213402, -115.6383743, 116.7939758
3: -73.2888107, 38.3888016, -76.6597824, 40.1798019, -113.4686127, 115.0485840
4: -67.1451111, 51.6868896, -70.3024368, 54.1350021, -121.2801132, 121.9893265
5: -60.0859833, 46.8676910, -62.8914833, 49.0638618, -109.1498413, 109.7591705
6: -58.0943642, 54.9306755, -60.7667923, 57.5371780, -115.6315308, 115.6974640
7: -63.1830101, 53.2205887, -66.1016159, 55.6565247, -118.8395081, 119.3222046
8: -74.8855972, 50.9028664, -78.4819489, 53.4345016, -128.3200836, 129.3848114
9: -57.5083313, 56.0916100, -60.2024956, 58.7882233, -116.2965317, 116.2941055

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3595426, upper bound: 197.3599680
time: 9.44 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3515979, upper bound: 197.3523210
time: 9.12 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -59.8200760, 47.6266594, -60.8697510, 48.4184799, -108.2385559, 108.4963913
1: -49.3474236, 42.1544495, -50.0153275, 42.7726173, -92.1200333, 92.1697769
2: -65.6871567, 43.3032761, -66.8051758, 43.9966698, -109.6838226, 110.1084442
3: -70.1833801, 36.7564964, -71.4478989, 37.3323898, -107.5157623, 108.2043839
4: -64.4210281, 49.7122459, -65.5340118, 50.4284515, -114.8494797, 115.2462463
5: -57.6207542, 45.0431480, -58.6323814, 45.7637749, -103.3845291, 103.6755295
6: -55.6669006, 52.6991234, -56.5990372, 53.5401230, -109.2070160, 109.2981491
7: -60.5179596, 51.0669212, -61.5266418, 51.8708763, -112.3888321, 112.5935669
8: -71.9067917, 48.9777641, -73.0490265, 49.6482468, -121.5550308, 122.0267868
9: -55.1552620, 53.7836685, -56.0138893, 54.5515633, -109.7068100, 109.7975540

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3686225, upper bound: 197.3682946
time: 10.22 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3835343, upper bound: 197.3834914
time: 65.07 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -69.7854080, 55.5617714, -70.2876511, 55.9518700, -125.7372742, 125.8494186
1: -57.9054909, 49.2735634, -58.2287292, 49.5797539, -107.4852219, 107.5022888
2: -76.6887131, 50.4037857, -77.2158585, 50.7315025, -127.4202118, 127.6196442
3: -81.7932053, 42.9311981, -82.3631210, 43.2098083, -125.0030060, 125.2943192
4: -75.1986237, 57.9975281, -75.7305298, 58.3538017, -133.5524292, 133.7280579
5: -67.1772156, 52.4559555, -67.6527176, 52.7972527, -119.9744720, 120.1086731
6: -64.8773956, 61.6117859, -65.3093643, 62.0186577, -126.8960495, 126.9211502
7: -70.5347137, 59.4456253, -71.0094910, 59.8069954, -130.3417053, 130.4551086
8: -84.1061935, 57.4502449, -84.6719513, 57.8251724, -141.9313507, 142.1221924
9: -64.3213425, 62.9046249, -64.7221832, 63.2976646, -127.6189957, 127.6268082

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4172329, upper bound: 197.4172422
time: 10.20 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4169111, upper bound: 197.4169853
time: 8.58 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -67.1603088, 53.3969421, -65.0828857, 51.7283401, -118.8886490, 118.4798126
1: -55.4135971, 47.2703781, -53.4964027, 45.7125511, -101.1261368, 100.7667770
2: -73.7326508, 48.4943123, -71.4241791, 46.9754982, -120.7081451, 119.9184799
3: -78.7637482, 41.2554893, -76.3735504, 39.9282036, -118.6919479, 117.6290436
4: -72.2851868, 55.6477280, -70.0397186, 53.8324471, -126.1176300, 125.6874390
5: -64.6203003, 50.4497490, -62.6497955, 48.8741150, -113.4944153, 113.0995178
6: -62.3595467, 59.1516228, -60.4224739, 57.2417145, -119.6012497, 119.5740967
7: -67.8795471, 57.1426773, -65.7542725, 55.3535957, -123.2331390, 122.8969498
8: -80.6998978, 54.9744263, -78.0848389, 53.0976639, -133.7975616, 133.0592651
9: -61.8293037, 60.3892212, -59.8488197, 58.3573341, -120.1866379, 120.2380371

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3654495, upper bound: 197.3654049
time: 9.84 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3656223, upper bound: 197.3656363
time: 9.28 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -77.3156967, 61.4853630, -74.7114105, 59.4296722, -136.7453613, 136.1967773
1: -64.1345673, 54.5246544, -61.8897858, 52.6685524, -116.8031158, 116.4144363
2: -84.9467087, 55.7254715, -82.0715485, 53.8584900, -138.8052063, 137.7969971
3: -90.5932159, 47.5500603, -87.5381699, 45.9349251, -136.5281372, 135.0882111
4: -83.2634583, 64.0939102, -80.4638519, 61.9348145, -145.1982574, 144.5577393
5: -74.3568726, 58.0026932, -71.8707657, 56.0622101, -130.4190826, 129.8734589
6: -71.7460327, 68.2343292, -69.3306656, 65.9106979, -137.6567383, 137.5649872
7: -78.0865097, 65.6803741, -75.4495468, 63.4671745, -141.5536804, 141.1299133
8: -93.1280212, 63.6089706, -89.9650726, 61.4530449, -154.5810699, 153.5740356
9: -71.1717377, 69.6852036, -68.7531204, 67.2962494, -138.4679871, 138.4383240

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4190989, upper bound: 197.4190507
time: 9.08 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4188422, upper bound: 197.4188837
time: 8.28 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -64.2052841, 51.0916100, -67.3019104, 53.5081329, -117.7134171, 118.3935089
1: -53.0855179, 45.2668724, -55.4883156, 47.3345718, -100.4200745, 100.7551880
2: -70.5269699, 46.4544678, -73.8977966, 48.6151962, -119.1421661, 120.3522644
3: -75.3283234, 39.4718323, -78.9910049, 41.3129234, -116.6412506, 118.4628372
4: -69.1393661, 53.3421478, -72.4506226, 55.7452736, -124.8846283, 125.7927704
5: -61.8141479, 48.3152962, -64.7933426, 50.5614967, -112.3756409, 113.1086349
6: -59.7106934, 56.5982018, -62.5304565, 59.2530632, -118.9637527, 119.1286469
7: -64.9546127, 54.7804947, -68.0282822, 57.3193550, -122.2739639, 122.8087692
8: -77.2231140, 52.6110039, -80.8457413, 54.9673691, -132.1904755, 133.4567108
9: -59.2365761, 57.8323135, -61.9892426, 60.4748840, -119.7114563, 119.8215561

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3671171, upper bound: 197.3670500
time: 8.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3674540, upper bound: 197.3674420
time: 8.98 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 19.22 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.2694769, upper bound: 197.2693909
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.2659924, upper bound: 197.2660871
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.3798935, upper bound: 197.3796932
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.3931427, upper bound: 197.3934385
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.3265362, upper bound: 197.3265268
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.3410271, upper bound: 197.3410494
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.3790868, upper bound: 197.3789717
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.3955439, upper bound: 197.3956275
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.2741080, upper bound: 197.2740022
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.2706947, upper bound: 197.2707914
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.3812977, upper bound: 197.3811039
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.3935476, upper bound: 197.3937971
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.3289167, upper bound: 197.3288545
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.3430761, upper bound: 197.3430823
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.3595426, upper bound: 197.3599680
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.3515979, upper bound: 197.3523210
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.3686225, upper bound: 197.3682946
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.3835343, upper bound: 197.3834914
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.4172329, upper bound: 197.4172422
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.4169111, upper bound: 197.4169853
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.3654495, upper bound: 197.3654049
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.3656223, upper bound: 197.3656363
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.4190989, upper bound: 197.4190507
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.4188422, upper bound: 197.4188837
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.3671171, upper bound: 197.3670500
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.22
Output dim: 4, lower bound: -197.3674540, upper bound: 197.3674420
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.22
Output dim: 4, lower bound: -197.4313008, upper bound: 197.4313171
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.22
Output dim: 4, lower bound: -197.3851721, upper bound: 197.3849824
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.22
Output dim: 4, lower bound: -197.4324535, upper bound: 197.4324535
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=198.953369140625
rel_dist={4: [-197.4407374020123, 197.4407374020123]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1830.62 seconds
