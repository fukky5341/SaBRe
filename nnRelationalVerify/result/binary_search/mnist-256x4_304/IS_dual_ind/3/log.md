## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 106.6602947382
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797)
1: (-48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317)
2: (-65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747)
3: (-68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471)
4: (-64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390)
5: (-57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115)
6: (-54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235)
7: (-58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829)
8: (-71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617)
9: (-54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888)

## BASE Result
execution time: IAR + LP analysis = 1.36 + 9.25 = 10.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -106.7671438, upper bound: 106.7671437


# Binary Search by BASE starts (time budget: 1989.39 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=108.02017974853516
rel_dist={0: [-106.7671031285499, 106.7671031285499]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=108.02017974853516
rel_dist={0: [-106.76706178863964, 106.76706178863964]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=108.02017974853516
rel_dist={0: [-106.76692036843552, 106.76692036843548]}

## Binary Search Result
Binary search time: 38.26 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1951.13 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7527244, upper bound: 106.7490664
time: 7.96 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7477250, upper bound: 106.7477250
time: 6.44 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.54 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 14.54
Output dim: 0, lower bound: -106.7527244, upper bound: 106.7490664
IS_A2, status: Status.UNKNOWN, split count: 1, time: 14.54
Output dim: 0, lower bound: -106.7477250, upper bound: 106.7477250

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -43.2367897, 34.4949570, -60.1304131, 47.8897781, -91.1265564, 94.6253510
1: -34.9878044, 29.8007889, -48.9951973, 41.5569344, -76.5447388, 78.7959671
2: -46.5671349, 30.1880875, -65.1370010, 42.1076736, -88.6748047, 95.3250885
3: -49.2782364, 25.9414082, -68.8342667, 36.2327805, -85.5110168, 94.7756729
4: -46.0794792, 34.7557182, -64.1016769, 48.5745659, -94.6540451, 98.8573914
5: -41.5383759, 32.7077026, -57.9381943, 45.3561211, -86.8945007, 90.6458969
6: -39.4550514, 37.8932304, -54.9252052, 52.6139221, -92.0689697, 92.8184280
7: -42.1981354, 36.3016472, -58.9155884, 50.4363098, -92.6344376, 95.2172241
8: -51.0103607, 33.9446335, -71.5229416, 47.6925201, -98.7028809, 105.4675751
9: -38.9905930, 38.0316658, -54.2685204, 52.9324722, -91.9230652, 92.3001785

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7477250, upper bound: 106.7477250
time: 6.57 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7477250, upper bound: 106.7477250
time: 51.38 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -35.7654915, 28.4981251, -54.3418922, 43.2901917, -79.0556793, 82.8400040
1: -28.6854954, 24.5347595, -44.1838303, 37.5126266, -66.1981201, 68.7185898
2: -38.3050308, 24.8694439, -58.7477226, 37.9755135, -76.2805481, 83.6171646
3: -40.5462570, 21.3767605, -62.1170464, 32.7000694, -73.2463226, 83.4938049
4: -37.9453735, 28.6490612, -57.8718605, 43.8203621, -81.7657318, 86.5209198
5: -34.1815300, 27.0604343, -52.3363266, 41.0165520, -75.1980820, 79.3967514
6: -32.5468407, 31.3406620, -49.5808792, 47.5468903, -80.0937119, 80.9215393
7: -34.7955933, 30.0198460, -53.1265182, 45.5785637, -80.3741455, 83.1463623
8: -41.8391533, 27.7593937, -64.4279175, 42.9504089, -84.7895660, 92.1873093
9: -32.1872482, 31.2693481, -49.0068626, 47.7884636, -79.9757080, 80.2762070

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7361704, upper bound: 106.7293898
time: 8.79 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435559, upper bound: 106.7435559
time: 7.98 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 18.07 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 18.07
Output dim: 0, lower bound: -106.7477250, upper bound: 106.7477250
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 18.07
Output dim: 0, lower bound: -106.7477250, upper bound: 106.7477250
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 18.07
Output dim: 0, lower bound: -106.7361704, upper bound: 106.7293898
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 18.07
Output dim: 0, lower bound: -106.7435559, upper bound: 106.7435559

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -43.2367897, 34.4949570, -43.2367897, 34.4949570, -77.7317276, 77.7317276
1: -34.9878044, 29.8007889, -34.9878044, 29.8007889, -64.7885895, 64.7885895
2: -46.5671349, 30.1880875, -46.5671349, 30.1880875, -76.7552185, 76.7552185
3: -49.2782364, 25.9414082, -49.2782364, 25.9414082, -75.2196426, 75.2196426
4: -46.0794792, 34.7557182, -46.0794792, 34.7557182, -80.8351974, 80.8351974
5: -41.5383759, 32.7077026, -41.5383759, 32.7077026, -74.2460709, 74.2460709
6: -39.4550514, 37.8932304, -39.4550514, 37.8932304, -77.3482742, 77.3482742
7: -42.1981354, 36.3016472, -42.1981354, 36.3016472, -78.4997787, 78.4997787
8: -51.0103607, 33.9446335, -51.0103607, 33.9446335, -84.9549942, 84.9549942
9: -38.9905930, 38.0316658, -38.9905930, 38.0316658, -77.0222549, 77.0222549

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7372876, upper bound: 106.7387138
time: 8.97 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7489573, upper bound: 106.7449337
time: 9.72 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -43.2367897, 34.4949570, -35.7654915, 28.4981251, -71.7349091, 70.2604370
1: -34.9878044, 29.8007889, -28.6854954, 24.5347595, -59.5225563, 58.4862823
2: -46.5671349, 30.1880875, -38.3050308, 24.8694439, -71.4365692, 68.4931183
3: -49.2782364, 25.9414082, -40.5462570, 21.3767605, -70.6549988, 66.4876633
4: -46.0794792, 34.7557182, -37.9453735, 28.6490612, -74.7285385, 72.7010956
5: -41.5383759, 32.7077026, -34.1815300, 27.0604343, -68.5987930, 66.8892365
6: -39.4550514, 37.8932304, -32.5468407, 31.3406620, -70.7957001, 70.4400558
7: -42.1981354, 36.3016472, -34.7955933, 30.0198460, -72.2179794, 71.0972443
8: -51.0103607, 33.9446335, -41.8391533, 27.7593937, -78.7697525, 75.7837830
9: -38.9905930, 38.0316658, -32.1872482, 31.2693481, -70.2599335, 70.2188950

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7372876, upper bound: 106.7387138
time: 9.77 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7489573, upper bound: 106.7449337
time: 8.88 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -33.6933861, 26.8429565, -39.0745735, 31.1763458, -64.8697357, 65.9175262
1: -26.9619675, 23.0907364, -31.4988270, 26.8993797, -53.8613434, 54.5895615
2: -36.0344315, 23.4065094, -42.0010376, 27.0360107, -63.0704422, 65.4075470
3: -38.1196442, 20.1037426, -44.3383751, 23.3906231, -61.5102692, 64.4421082
4: -35.7070198, 26.9795094, -41.4685364, 31.3803539, -67.0873718, 68.4480438
5: -32.1628799, 25.4967918, -37.5362206, 29.6096001, -61.7724800, 63.0330009
6: -30.6350803, 29.5311012, -35.4811134, 34.2977600, -64.9328384, 65.0122147
7: -32.7627983, 28.2730331, -37.9530487, 32.7629242, -65.5257263, 66.2260666
8: -39.3324814, 26.0907555, -45.7810326, 30.4865589, -69.8190384, 71.8717804
9: -30.3170700, 29.4214077, -35.2016144, 34.1691895, -64.4862595, 64.6230164

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7256890, upper bound: 106.7173637
time: 8.86 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7245304, upper bound: 106.7170019
time: 8.42 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -35.7654915, 28.4981251, -47.1821899, 37.6130638, -73.3785553, 75.6803131
1: -28.6854954, 24.5347595, -38.2502747, 32.5439453, -61.2294388, 62.7850266
2: -38.3050308, 24.8694439, -50.9027748, 32.8385811, -71.1436157, 75.7722015
3: -40.5462570, 21.3767605, -53.7999535, 28.3437023, -68.8899612, 75.1767120
4: -37.9453735, 28.6490612, -50.2027359, 37.9690514, -75.9144287, 78.8517914
5: -34.1815300, 27.0604343, -45.4262390, 35.6650162, -69.8465424, 72.4866562
6: -32.5468407, 31.3406620, -42.9736710, 41.3295860, -73.8764191, 74.3143311
7: -34.7955933, 30.0198460, -45.9976158, 39.5672531, -74.3628464, 76.0174408
8: -41.8391533, 27.7593937, -55.6884766, 37.0967369, -78.9358902, 83.4478683
9: -32.1872482, 31.2693481, -42.5228691, 41.4217377, -73.6089630, 73.7921906

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7293898, upper bound: 106.7361704
time: 8.56 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7293898, upper bound: 106.7435559
time: 8.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.32 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 0, lower bound: -106.7372876, upper bound: 106.7387138
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 0, lower bound: -106.7489573, upper bound: 106.7449337
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 0, lower bound: -106.7372876, upper bound: 106.7387138
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 0, lower bound: -106.7489573, upper bound: 106.7449337
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 0, lower bound: -106.7256890, upper bound: 106.7173637
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 0, lower bound: -106.7245304, upper bound: 106.7170019
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 0, lower bound: -106.7293898, upper bound: 106.7361704
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 0, lower bound: -106.7293898, upper bound: 106.7435559

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -30.0412292, 23.9613342, -40.8639870, 32.6135635, -62.6547928, 64.8253174
1: -24.0112438, 20.6137333, -33.0195045, 28.1583061, -52.1695480, 53.6332359
2: -32.1210938, 20.7836094, -43.9688873, 28.4998856, -60.6209717, 64.7524948
3: -33.8338737, 17.8628426, -46.5151558, 24.4987965, -58.3326645, 64.3779984
4: -31.8047161, 24.1073666, -43.5212746, 32.8340836, -64.6388016, 67.6286392
5: -28.6844864, 22.7719288, -39.2288322, 30.9340267, -59.6185150, 62.0007629
6: -27.2614193, 26.4016151, -37.2710495, 35.8344994, -63.0959167, 63.6726570
7: -29.1837025, 25.2196217, -39.8497505, 34.3166351, -63.5003357, 65.0693665
8: -34.9649467, 23.2314949, -48.1173096, 32.0097046, -66.9746552, 71.3487930
9: -27.0878944, 26.2169075, -36.8542633, 35.9158630, -63.0037575, 63.0711670

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7298775, upper bound: 106.7368202
time: 9.30 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7294585, upper bound: 106.7354458
time: 8.14 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -36.8479729, 29.4056492, -43.2367897, 34.4949570, -71.3429260, 72.6424332
1: -29.6894016, 25.3731251, -34.9878044, 29.8007889, -59.4901886, 60.3609276
2: -39.5855675, 25.6514072, -46.5671349, 30.1880875, -69.7736511, 72.2185440
3: -41.8327026, 22.0612946, -49.2782364, 25.9414082, -67.7741089, 71.3395309
4: -39.1989174, 29.5916195, -46.0794792, 34.7557182, -73.9546356, 75.6710968
5: -35.3177071, 27.9220924, -41.5383759, 32.7077026, -68.0254059, 69.4604568
6: -33.5822754, 32.3391953, -39.4550514, 37.8932304, -71.4755096, 71.7942352
7: -35.8931923, 30.9565773, -42.1981354, 36.3016472, -72.1948395, 73.1547089
8: -43.2296486, 28.7269249, -51.0103607, 33.9446335, -77.1742859, 79.7372894
9: -33.2405930, 32.3367348, -38.9905930, 38.0316658, -71.2722397, 71.3273315

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7474763, upper bound: 106.7433446
time: 8.70 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7474763, upper bound: 106.7433446
time: 8.41 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -30.0412292, 23.9613342, -33.6933861, 26.8429565, -56.8841782, 57.6547203
1: -24.0112438, 20.6137333, -26.9619675, 23.0907364, -47.1019783, 47.5756950
2: -32.1210938, 20.7836094, -36.0344315, 23.4065094, -55.5276031, 56.8180389
3: -33.8338737, 17.8628426, -38.1196442, 20.1037426, -53.9376144, 55.9824791
4: -31.8047161, 24.1073666, -35.7070198, 26.9795094, -58.7842216, 59.8143845
5: -28.6844864, 22.7719288, -32.1628799, 25.4967918, -54.1812744, 54.9348068
6: -27.2614193, 26.4016151, -30.6350803, 29.5311012, -56.7925186, 57.0366936
7: -29.1837025, 25.2196217, -32.7627983, 28.2730331, -57.4567337, 57.9824142
8: -34.9649467, 23.2314949, -39.3324814, 26.0907555, -61.0557022, 62.5639725
9: -27.0878944, 26.2169075, -30.3170700, 29.4214077, -56.5093002, 56.5339737

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7243801, upper bound: 106.7277367
time: 8.03 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7240767, upper bound: 106.7266918
time: 9.51 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -36.8479729, 29.4056492, -35.7654915, 28.4981251, -65.3460999, 65.1711426
1: -29.6894016, 25.3731251, -28.6854954, 24.5347595, -54.2241554, 54.0586205
2: -39.5855675, 25.6514072, -38.3050308, 24.8694439, -64.4550095, 63.9564362
3: -41.8327026, 22.0612946, -40.5462570, 21.3767605, -63.2094650, 62.6075516
4: -39.1989174, 29.5916195, -37.9453735, 28.6490612, -67.8479767, 67.5369949
5: -35.3177071, 27.9220924, -34.1815300, 27.0604343, -62.3781395, 62.1036148
6: -33.5822754, 32.3391953, -32.5468407, 31.3406620, -64.9229355, 64.8860168
7: -35.8931923, 30.9565773, -34.7955933, 30.0198460, -65.9130325, 65.7521667
8: -43.2296486, 28.7269249, -41.8391533, 27.7593937, -70.9890442, 70.5660782
9: -33.2405930, 32.3367348, -32.1872482, 31.2693481, -64.5099258, 64.5239716

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7406225, upper bound: 106.7307550
time: 7.79 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7406225, upper bound: 106.7449337
time: 7.92 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -22.7261448, 18.0841331, -39.0745735, 31.1763458, -53.9024887, 57.1587029
1: -17.8482761, 15.5313196, -31.4988270, 26.8993797, -44.7476463, 47.0301476
2: -24.0871620, 15.7524643, -42.0010376, 27.0360107, -51.1231728, 57.7535019
3: -25.4113750, 13.4992256, -44.3383751, 23.3906231, -48.8019943, 57.8375969
4: -23.9650574, 18.1435032, -41.4685364, 31.3803539, -55.3454132, 59.6120338
5: -21.4803963, 17.2252636, -37.5362206, 29.6096001, -51.0899963, 54.7614784
6: -20.6362267, 19.9110870, -35.4811134, 34.2977600, -54.9339867, 55.3922005
7: -22.0875759, 19.1069584, -37.9530487, 32.7629242, -54.8504982, 57.0600052
8: -26.2941284, 17.4104977, -45.7810326, 30.4865589, -56.7806854, 63.1915283
9: -20.4198685, 19.7437458, -35.2016144, 34.1691895, -54.5890579, 54.9453583

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7245303, upper bound: 106.7169917
time: 9.08 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7245303, upper bound: 106.7170016
time: 9.00 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -23.0983105, 18.3585472, -36.3415604, 28.9834652, -52.0817757, 54.7001076
1: -18.1202946, 15.7661800, -29.2209072, 24.9869690, -43.1072540, 44.9870834
2: -24.4671669, 15.9701424, -38.9915504, 25.1099186, -49.5770798, 54.9616928
3: -25.8341446, 13.7052946, -41.1523552, 21.7151566, -47.5492973, 54.8576508
4: -24.3241386, 18.4411335, -38.5154076, 29.1678734, -53.4920044, 56.9565430
5: -21.8190708, 17.4581623, -34.8718414, 27.5589924, -49.3780632, 52.3299942
6: -20.9549618, 20.2402954, -32.9659424, 31.8992729, -52.8542328, 53.2062378
7: -22.4447651, 19.3911858, -35.2679977, 30.4835224, -52.9282875, 54.6591835
8: -26.7129345, 17.6762695, -42.4455833, 28.2442284, -54.9571609, 60.1218529
9: -20.7368355, 20.0565968, -32.7316704, 31.7085476, -52.4453812, 52.7882652

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7155197, upper bound: 106.7099682
time: 8.85 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7149950, upper bound: 106.7076148
time: 8.34 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -24.4317169, 19.4307880, -47.1821899, 37.6130638, -62.0447807, 66.6129684
1: -19.2368259, 16.6793652, -38.2502747, 32.5439453, -51.7807655, 54.9296417
2: -25.9408703, 16.8350620, -50.9027748, 32.8385811, -58.7794495, 67.7378235
3: -27.2833900, 14.4726753, -53.7999535, 28.3437023, -55.6270905, 68.2726135
4: -25.7207546, 19.5351772, -50.2027359, 37.9690514, -63.6898041, 69.7379150
5: -23.1219635, 18.4612331, -45.4262390, 35.6650162, -58.7869797, 63.8874741
6: -22.0781078, 21.4634705, -42.9736710, 41.3295860, -63.4076920, 64.4371414
7: -23.6715374, 20.4631119, -45.9976158, 39.5672531, -63.2387924, 66.4607162
8: -28.2296677, 18.7803955, -55.6884766, 37.0967369, -65.3264008, 74.4688568
9: -21.9435825, 21.1932259, -42.5228691, 41.4217377, -63.3653183, 63.7160912

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7147760, upper bound: 106.7256889
time: 8.27 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7145331, upper bound: 106.7245304
time: 8.50 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -30.3078403, 24.1355877, -47.1821899, 37.6130638, -67.9209061, 71.3177719
1: -24.1456814, 20.7440796, -38.2502747, 32.5439453, -56.6896210, 58.9943542
2: -32.3482323, 21.0194035, -50.9027748, 32.8385811, -65.1868134, 71.9221802
3: -34.1637001, 18.0307159, -53.7999535, 28.3437023, -62.5074005, 71.8306732
4: -32.0680580, 24.2503166, -50.2027359, 37.9690514, -70.0371094, 74.4530411
5: -28.8591881, 22.9372196, -45.4262390, 35.6650162, -64.5242004, 68.3634415
6: -27.5124569, 26.5735931, -42.9736710, 41.3295860, -68.8420410, 69.5472641
7: -29.4583912, 25.4157600, -45.9976158, 39.5672531, -69.0256424, 71.4133530
8: -35.2714806, 23.3929329, -55.6884766, 37.0967369, -72.3682175, 79.0813904
9: -27.2663784, 26.4264870, -42.5228691, 41.4217377, -68.6881027, 68.9493408

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7147760, upper bound: 106.7336298
time: 8.84 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7145331, upper bound: 106.7145330
time: 7.98 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 18.17 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.7298775, upper bound: 106.7368202
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.7294585, upper bound: 106.7354458
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.7474763, upper bound: 106.7433446
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.7474763, upper bound: 106.7433446
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.7243801, upper bound: 106.7277367
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.7240767, upper bound: 106.7266918
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.7406225, upper bound: 106.7307550
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.7406225, upper bound: 106.7449337
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.7245303, upper bound: 106.7169917
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.7245303, upper bound: 106.7170016
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.7155197, upper bound: 106.7099682
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.7149950, upper bound: 106.7076148
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.7147760, upper bound: 106.7256889
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.7145331, upper bound: 106.7245304
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.7147760, upper bound: 106.7336298
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.7145331, upper bound: 106.7145330

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -30.0412292, 23.9613342, -28.4065685, 22.6621914, -52.7034225, 52.3679047
1: -24.0112438, 20.6137333, -22.6932030, 19.5110683, -43.5223122, 43.3069267
2: -32.1210938, 20.7836094, -30.3695583, 19.7759838, -51.8970757, 51.1531677
3: -33.8338737, 17.8628426, -32.0442009, 16.9283314, -50.7622070, 49.9070320
4: -31.8047161, 24.1073666, -30.1588345, 22.7872944, -54.5920105, 54.2661934
5: -28.6844864, 22.7719288, -27.0826416, 21.5762367, -50.2607231, 49.8545685
6: -27.2614193, 26.4016151, -25.8775158, 24.9331837, -52.1946030, 52.2791214
7: -29.1837025, 25.2196217, -27.6799183, 23.9340916, -53.1177940, 52.8995323
8: -34.9649467, 23.2314949, -33.1265793, 21.9147282, -56.8796768, 56.3580742
9: -27.0878944, 26.2169075, -25.6399841, 24.8404922, -51.9283867, 51.8568840

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7294585, upper bound: 106.7354458
time: 7.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7294585, upper bound: 106.7354458
time: 7.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -27.6694164, 22.0676193, -27.5760860, 21.9754772, -49.6448936, 49.6436996
1: -22.0304012, 18.9629955, -21.9672031, 18.9107800, -40.9411812, 40.9301987
2: -29.5310440, 19.1234932, -29.4331150, 19.1535988, -48.6846390, 48.5566063
3: -31.0668697, 16.4224949, -31.0710392, 16.4064369, -47.4733047, 47.4935341
4: -29.2520924, 22.1921177, -29.2221642, 22.1211281, -51.3732224, 51.4142838
5: -26.3641930, 20.9842205, -26.2530670, 20.9046993, -47.2688904, 47.2372894
6: -25.0818558, 24.3149605, -25.0857201, 24.2052326, -49.2870865, 49.4006805
7: -26.8663769, 23.2318096, -26.8550663, 23.2176342, -50.0840111, 50.0868759
8: -32.1366272, 21.3476734, -32.1016922, 21.2255421, -53.3621635, 53.4493637
9: -24.9393578, 24.1121292, -24.8681545, 24.0800705, -49.0194168, 48.9802856

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7209432, upper bound: 106.7242645
time: 7.47 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7162846, upper bound: 106.7231705
time: 8.32 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -36.8479729, 29.4056492, -30.0412292, 23.9613342, -60.8093071, 59.4468765
1: -29.6894016, 25.3731251, -24.0112438, 20.6137333, -50.3031349, 49.3843651
2: -39.5855675, 25.6514072, -32.1210938, 20.7836094, -60.3691788, 57.7724991
3: -41.8327026, 22.0612946, -33.8338737, 17.8628426, -59.6955376, 55.8951683
4: -39.1989174, 29.5916195, -31.8047161, 24.1073666, -63.3062820, 61.3963318
5: -35.3177071, 27.9220924, -28.6844864, 22.7719288, -58.0896378, 56.6065712
6: -33.5822754, 32.3391953, -27.2614193, 26.4016151, -59.9838829, 59.6006165
7: -35.8931923, 30.9565773, -29.1837025, 25.2196217, -61.1128044, 60.1402817
8: -43.2296486, 28.7269249, -34.9649467, 23.2314949, -66.4611359, 63.6918716
9: -33.2405930, 32.3367348, -27.0878944, 26.2169075, -59.4574928, 59.4246292

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7368202, upper bound: 106.7298775
time: 8.03 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7354458, upper bound: 106.7294585
time: 9.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -36.8479729, 29.4056492, -36.8479729, 29.4056492, -66.2536240, 66.2536240
1: -29.6894016, 25.3731251, -29.6894016, 25.3731251, -55.0625267, 55.0625267
2: -39.5855675, 25.6514072, -39.5855675, 25.6514072, -65.2369766, 65.2369766
3: -41.8327026, 22.0612946, -41.8327026, 22.0612946, -63.8939972, 63.8939972
4: -39.1989174, 29.5916195, -39.1989174, 29.5916195, -68.7905350, 68.7905350
5: -35.3177071, 27.9220924, -35.3177071, 27.9220924, -63.2397919, 63.2397919
6: -33.5822754, 32.3391953, -33.5822754, 32.3391953, -65.9214706, 65.9214706
7: -35.8931923, 30.9565773, -35.8931923, 30.9565773, -66.8497696, 66.8497696
8: -43.2296486, 28.7269249, -43.2296486, 28.7269249, -71.9565735, 71.9565735
9: -33.2405930, 32.3367348, -33.2405930, 32.3367348, -65.5773163, 65.5773163

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7368202, upper bound: 106.7430474
time: 8.19 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7354458, upper bound: 106.7426484
time: 10.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -30.0412292, 23.9613342, -22.7261448, 18.0841331, -48.1253586, 46.6874771
1: -24.0112438, 20.6137333, -17.8482761, 15.5313196, -39.5425644, 38.4619980
2: -32.1210938, 20.7836094, -24.0871620, 15.7524643, -47.8735542, 44.8707733
3: -33.8338737, 17.8628426, -25.4113750, 13.4992256, -47.3330956, 43.2742081
4: -31.8047161, 24.1073666, -23.9650574, 18.1435032, -49.9482117, 48.0724220
5: -28.6844864, 22.7719288, -21.4803963, 17.2252636, -45.9097519, 44.2523270
6: -27.2614193, 26.4016151, -20.6362267, 19.9110870, -47.1725082, 47.0378342
7: -29.1837025, 25.2196217, -22.0875759, 19.1069584, -48.2906609, 47.3071899
8: -34.9649467, 23.2314949, -26.2941284, 17.4104977, -52.3754425, 49.5256195
9: -27.0878944, 26.2169075, -20.4198685, 19.7437458, -46.8316422, 46.6367722

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7240767, upper bound: 106.7266919
time: 8.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7240767, upper bound: 106.7266919
time: 9.10 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -27.6694164, 22.0676193, -23.0983105, 18.3585472, -46.0279617, 45.1659317
1: -22.0304012, 18.9629955, -18.1202946, 15.7661800, -37.7965813, 37.0832863
2: -29.5310440, 19.1234932, -24.4671669, 15.9701424, -45.5011787, 43.5906563
3: -31.0668697, 16.4224949, -25.8341446, 13.7052946, -44.7721634, 42.2566376
4: -29.2520924, 22.1921177, -24.3241386, 18.4411335, -47.6932182, 46.5162582
5: -26.3641930, 20.9842205, -21.8190708, 17.4581623, -43.8223534, 42.8032913
6: -25.0818558, 24.3149605, -20.9549618, 20.2402954, -45.3221436, 45.2699203
7: -26.8663769, 23.2318096, -22.4447651, 19.3911858, -46.2575607, 45.6765747
8: -32.1366272, 21.3476734, -26.7129345, 17.6762695, -49.8128929, 48.0606079
9: -24.9393578, 24.1121292, -20.7368355, 20.0565968, -44.9959450, 44.8489647

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7169945, upper bound: 106.7177640
time: 8.47 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7125614, upper bound: 106.7166934
time: 8.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -36.8479729, 29.4056492, -24.4317169, 19.4307880, -56.2787628, 53.8373642
1: -29.6894016, 25.3731251, -19.2368259, 16.6793652, -46.3687668, 44.6099472
2: -39.5855675, 25.6514072, -25.9408703, 16.8350620, -56.4206314, 51.5922775
3: -41.8327026, 22.0612946, -27.2833900, 14.4726753, -56.3053741, 49.3446808
4: -39.1989174, 29.5916195, -25.7207546, 19.5351772, -58.7340927, 55.3123665
5: -35.3177071, 27.9220924, -23.1219635, 18.4612331, -53.7789383, 51.0440559
6: -33.5822754, 32.3391953, -22.0781078, 21.4634705, -55.0457344, 54.4173050
7: -35.8931923, 30.9565773, -23.6715374, 20.4631119, -56.3563042, 54.6281128
8: -43.2296486, 28.7269249, -28.2296677, 18.7803955, -62.0100441, 56.9565926
9: -33.2405930, 32.3367348, -21.9435825, 21.1932259, -54.4338150, 54.2803192

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7306722, upper bound: 106.7189690
time: 7.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7294651, upper bound: 106.7186073
time: 9.34 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -36.8479729, 29.4056492, -30.3078403, 24.1355877, -60.9835587, 59.7134895
1: -29.6894016, 25.3731251, -24.1456814, 20.7440796, -50.4334793, 49.5188026
2: -39.5855675, 25.6514072, -32.3482323, 21.0194035, -60.6049690, 57.9996414
3: -41.8327026, 22.0612946, -34.1637001, 18.0307159, -59.8634186, 56.2249908
4: -39.1989174, 29.5916195, -32.0680580, 24.2503166, -63.4492340, 61.6596756
5: -35.3177071, 27.9220924, -28.8591881, 22.9372196, -58.2549210, 56.7812805
6: -33.5822754, 32.3391953, -27.5124569, 26.5735931, -60.1558685, 59.8516541
7: -35.8931923, 30.9565773, -29.4583912, 25.4157600, -61.3089523, 60.4149704
8: -43.2296486, 28.7269249, -35.2714806, 23.3929329, -66.6225662, 63.9984055
9: -33.2405930, 32.3367348, -27.2663784, 26.4264870, -59.6670799, 59.6031113

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7306722, upper bound: 106.7338351
time: 9.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7294651, upper bound: 106.7334685
time: 8.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -22.7261448, 18.0841331, -26.6738796, 21.2779808, -44.0041237, 44.7580109
1: -17.8482761, 15.5313196, -21.2304993, 18.2825985, -36.1308708, 36.7618179
2: -24.0871620, 15.7524643, -28.4572029, 18.3825912, -42.4697533, 44.2096672
3: -25.4113750, 13.4992256, -29.9255886, 15.8133831, -41.2247581, 43.4248085
4: -23.9650574, 18.1435032, -28.1803093, 21.3853951, -45.3504524, 46.3238144
5: -21.4803963, 17.2252636, -25.4515724, 20.2873840, -41.7677803, 42.6768303
6: -20.6362267, 19.9110870, -24.1462097, 23.4439735, -44.0801926, 44.0572929
7: -22.0875759, 19.1069584, -25.8756618, 22.4156036, -44.5031776, 44.9826202
8: -26.2941284, 17.4104977, -30.9073257, 20.4948921, -46.7890205, 48.3178253
9: -20.4198685, 19.7437458, -24.0300694, 23.1765823, -43.5964470, 43.7738152

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7237701, upper bound: 106.7155345
time: 8.76 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7239019, upper bound: 106.7155638
time: 8.82 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -22.7261448, 18.0841331, -26.5712395, 21.1747360, -43.9008751, 44.6553650
1: -17.8482761, 15.5313196, -21.1108341, 18.1902180, -36.0384789, 36.6421547
2: -24.0871620, 15.7524643, -28.3187141, 18.2651291, -42.3522911, 44.0711784
3: -25.4113750, 13.4992256, -29.8117523, 15.7316074, -41.1429825, 43.3109703
4: -23.9650574, 18.1435032, -28.0174522, 21.3101921, -45.2752495, 46.1609497
5: -21.4803963, 17.2252636, -25.3364353, 20.1683655, -41.6487579, 42.5616989
6: -20.6362267, 19.9110870, -24.0297089, 23.3570309, -43.9932556, 43.9407921
7: -22.0875759, 19.1069584, -25.7517719, 22.3016186, -44.3891907, 44.8587303
8: -26.2941284, 17.4104977, -30.7524910, 20.3874702, -46.6815948, 48.1629868
9: -20.4198685, 19.7437458, -23.9108143, 23.0601730, -43.4800377, 43.6545601

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7237701, upper bound: 106.7155345
time: 7.91 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7239019, upper bound: 106.7155638
time: 7.16 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -23.0983105, 18.3585472, -29.8015938, 23.8564682, -46.9547729, 48.1601410
1: -18.1202946, 15.7661800, -23.9134960, 20.5323792, -38.6526680, 39.6796722
2: -24.4671669, 15.9701424, -31.9534512, 20.6352634, -45.1024246, 47.9235916
3: -25.8341446, 13.7052946, -33.6783028, 17.7915382, -43.6256828, 47.3835983
4: -24.3241386, 18.4411335, -31.6435604, 24.0012245, -48.3253632, 50.0846939
5: -21.8190708, 17.4581623, -28.5806293, 22.7589073, -44.5779800, 46.0387802
6: -20.9549618, 20.2402954, -27.0712013, 26.2164841, -47.1714478, 47.3114891
7: -22.4447651, 19.3911858, -29.0182610, 25.1259480, -47.5707130, 48.4094429
8: -26.7129345, 17.6762695, -34.7296791, 23.1235352, -49.8364677, 52.4059486
9: -20.7368355, 20.0565968, -27.0015335, 26.0933743, -46.8302078, 47.0581284

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7137672, upper bound: 106.7081337
time: 8.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7136811, upper bound: 106.7081280
time: 8.89 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -21.8957691, 17.4263020, -31.5034256, 25.2132225, -47.1089897, 48.9297142
1: -17.1335011, 14.9471560, -25.3474197, 21.7091713, -38.8426743, 40.2945747
2: -23.1784401, 15.1487131, -33.8180580, 21.8803177, -45.0587578, 48.9667702
3: -24.4636745, 12.9832277, -35.6712456, 18.8445797, -43.3082504, 48.6544724
4: -23.0514164, 17.4962788, -33.5336800, 25.3580608, -48.4094772, 51.0299606
5: -20.6707306, 16.5727882, -30.2242851, 24.0511036, -44.7218323, 46.7970695
6: -19.8839149, 19.1891232, -28.6851330, 27.6974888, -47.5814018, 47.8742561
7: -21.2864113, 18.4060326, -30.7449913, 26.6079884, -47.8944016, 49.1510239
8: -25.3193359, 16.7674351, -36.7861786, 24.5122681, -49.8316040, 53.5536079
9: -19.6846294, 19.0206223, -28.5789795, 27.6972637, -47.3818932, 47.5995979

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7132602, upper bound: 106.7057039
time: 8.22 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7131691, upper bound: 106.7056963
time: 8.35 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -24.4317169, 19.4307880, -33.1699791, 26.4506760, -50.8823929, 52.6007690
1: -19.2368259, 16.6793652, -26.6283665, 22.8071461, -42.0439682, 43.3077316
2: -25.9408703, 16.8350620, -35.5563736, 22.9969616, -48.9378319, 52.3914337
3: -27.2833900, 14.4726753, -37.5459518, 19.8000565, -47.0834427, 52.0186234
4: -25.7207546, 19.5351772, -35.2050209, 26.5985527, -52.3193016, 54.7401962
5: -23.1219635, 18.4612331, -31.7715168, 25.1892738, -48.3112373, 50.2327499
6: -22.0781078, 21.4634705, -30.1481667, 29.0993347, -51.1774445, 51.6116257
7: -23.6715374, 20.4631119, -32.2496147, 27.8853207, -51.5568581, 52.7127266
8: -28.2296677, 18.7803955, -38.7016335, 25.6555347, -53.8852005, 57.4820290
9: -21.9435825, 21.1932259, -29.8968506, 28.9556599, -50.8992386, 51.0900764

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7103933, upper bound: 106.7170075
time: 9.08 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7080396, upper bound: 106.7165244
time: 7.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -22.6009102, 17.9719830, -32.9586754, 26.2627583, -48.8636703, 50.9306564
1: -17.7045574, 15.4180346, -26.4247417, 22.6367149, -40.3412704, 41.8427773
2: -23.9469814, 15.5501356, -35.3080254, 22.8010998, -46.7480774, 50.8581619
3: -25.1720066, 13.3738184, -37.3155022, 19.6432381, -44.8152466, 50.6893196
4: -23.7511177, 18.0672722, -34.9458313, 26.4397030, -50.1908150, 53.0131035
5: -21.3443127, 17.0866890, -31.5583096, 24.9885349, -46.3328400, 48.6449966
6: -20.4261150, 19.8503742, -29.9305553, 28.9194679, -49.3455811, 49.7809296
7: -21.8785286, 18.9265747, -32.0418053, 27.6882477, -49.5667763, 50.9683762
8: -26.0632572, 17.3420849, -38.3990250, 25.4421673, -51.5054245, 55.7411041
9: -20.2864437, 19.5648975, -29.6975803, 28.7389641, -49.0254059, 49.2624779

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7153211, upper bound: 106.7226988
time: 8.39 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7151789, upper bound: 106.7226594
time: 8.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -30.3078403, 24.1355877, -33.1699791, 26.4506760, -56.7585144, 57.3055649
1: -24.1456814, 20.7440796, -26.6283665, 22.8071461, -46.9528236, 47.3724403
2: -32.3482323, 21.0194035, -35.5563736, 22.9969616, -55.3451920, 56.5757751
3: -34.1637001, 18.0307159, -37.5459518, 19.8000565, -53.9637527, 55.5766678
4: -32.0680580, 24.2503166, -35.2050209, 26.5985527, -58.6666107, 59.4553337
5: -28.8591881, 22.9372196, -31.7715168, 25.1892738, -54.0484619, 54.7087326
6: -27.5124569, 26.5735931, -30.1481667, 29.0993347, -56.6117935, 56.7217598
7: -29.4583912, 25.4157600, -32.2496147, 27.8853207, -57.3437080, 57.6653748
8: -35.2714806, 23.3929329, -38.7016335, 25.6555347, -60.9270096, 62.0945663
9: -27.2663784, 26.4264870, -29.8968506, 28.9556599, -56.2220345, 56.3233376

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 109

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319790, upper bound: 106.7319806
time: 7.55 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319790, upper bound: 106.7319806
time: 7.13 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 16.05 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7294585, upper bound: 106.7354458
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7294585, upper bound: 106.7354458
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7209432, upper bound: 106.7242645
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7162846, upper bound: 106.7231705
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7368202, upper bound: 106.7298775
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7354458, upper bound: 106.7294585
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7368202, upper bound: 106.7430474
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7354458, upper bound: 106.7426484
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7240767, upper bound: 106.7266919
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7240767, upper bound: 106.7266919
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7169945, upper bound: 106.7177640
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7125614, upper bound: 106.7166934
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7306722, upper bound: 106.7189690
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7294651, upper bound: 106.7186073
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7306722, upper bound: 106.7338351
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7294651, upper bound: 106.7334685
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7237701, upper bound: 106.7155345
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7239019, upper bound: 106.7155638
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7237701, upper bound: 106.7155345
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7239019, upper bound: 106.7155638
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7137672, upper bound: 106.7081337
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7136811, upper bound: 106.7081280
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7132602, upper bound: 106.7057039
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7131691, upper bound: 106.7056963
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7103933, upper bound: 106.7170075
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7080396, upper bound: 106.7165244
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7153211, upper bound: 106.7226988
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7151789, upper bound: 106.7226594
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7319790, upper bound: 106.7319806
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.05
Output dim: 0, lower bound: -106.7319790, upper bound: 106.7319806
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.05
Output dim: 0, lower bound: -106.7145331, upper bound: 106.7145330
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=108.02017974853516
rel_dist={0: [-106.7671031285499, 106.7671031285499]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7509480, upper bound: 106.7484515
time: 8.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7477160, upper bound: 106.7477160
time: 8.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.68 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.68
Output dim: 0, lower bound: -106.7509480, upper bound: 106.7484515
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.68
Output dim: 0, lower bound: -106.7477160, upper bound: 106.7477160

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -43.2367897, 34.4949570, -57.1260300, 45.5063477, -88.7431259, 91.6209717
1: -34.9878044, 29.8007889, -46.5087395, 39.4619522, -74.4497528, 76.3095169
2: -46.5671349, 30.1880875, -61.8263855, 39.9733696, -86.5405045, 92.0144577
3: -49.2782364, 25.9414082, -65.3586578, 34.4015312, -83.6797638, 91.3000641
4: -46.0794792, 34.7557182, -60.8894615, 46.1131668, -92.1926422, 95.6451797
5: -41.5383759, 32.7077026, -55.0305443, 43.1037903, -84.6421432, 87.7382431
6: -39.4550514, 37.8932304, -52.1641197, 49.9888725, -89.4439163, 90.0573502
7: -42.1981354, 36.3016472, -55.9273071, 47.9200172, -90.1181335, 92.2289581
8: -51.0103607, 33.9446335, -67.8608627, 45.2444916, -96.2548523, 101.8054886
9: -38.9905930, 38.0316658, -51.5498238, 50.2817078, -89.2722931, 89.5814896

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7404419, upper bound: 106.7383882
time: 9.08 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7403021, upper bound: 106.7374048
time: 10.00 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -35.7654915, 28.4981251, -46.8274879, 37.3221436, -73.0876312, 75.3256149
1: -28.6854954, 24.5347595, -37.9242325, 32.2664948, -60.9519882, 62.4589844
2: -38.3050308, 24.8694439, -50.4714355, 32.6462212, -70.9512482, 75.3408661
3: -40.5462570, 21.3767605, -53.3877144, 28.1204891, -68.6667480, 74.7644730
4: -37.9453735, 28.6490612, -49.8108444, 37.6521950, -75.5975647, 78.4599075
5: -34.1815300, 27.0604343, -45.0441284, 35.3928070, -69.5743256, 72.1045532
6: -32.5468407, 31.3406620, -42.6719742, 40.9815521, -73.5283813, 74.0126343
7: -34.7955933, 30.0198460, -45.6488113, 39.2752914, -74.0708771, 75.6686554
8: -41.8391533, 27.7593937, -55.2702980, 36.7987862, -78.6379395, 83.0296936
9: -32.1872482, 31.2693481, -42.1782913, 41.1054382, -73.2926865, 73.4476242

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7329482, upper bound: 106.7281844
time: 9.88 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
time: 8.15 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.37 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 19.37
Output dim: 0, lower bound: -106.7404419, upper bound: 106.7383882
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 19.37
Output dim: 0, lower bound: -106.7403021, upper bound: 106.7374048
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 19.37
Output dim: 0, lower bound: -106.7329482, upper bound: 106.7281844
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 19.37
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -39.4144402, 31.4517403, -41.7696991, 33.2865639, -72.7010040, 73.2214355
1: -31.8218117, 27.1506996, -33.7678795, 28.7669888, -60.5887947, 60.9185791
2: -42.3857384, 27.4988995, -44.9783859, 29.0818100, -71.4675369, 72.4772873
3: -44.8457909, 23.6152859, -47.5543365, 25.0169697, -69.8627548, 71.1696243
4: -41.9799309, 31.6541100, -44.4619865, 33.5388527, -75.5187683, 76.1160965
5: -37.8115425, 29.8467808, -40.1380577, 31.6360874, -69.4476166, 69.9848404
6: -35.9555817, 34.5581627, -38.0400887, 36.5754623, -72.5310364, 72.5982513
7: -38.4419250, 33.1212234, -40.7144432, 35.0862427, -73.5281677, 73.8356628
8: -46.3705750, 30.8078651, -49.1352615, 32.6204224, -78.9909973, 79.9431305
9: -35.5475807, 34.6308556, -37.6220818, 36.6176987, -72.1652679, 72.2529373

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7205639, upper bound: 106.7228340
time: 8.77 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7361426, upper bound: 106.7339479
time: 8.71 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -35.5550346, 28.3686314, -41.0954018, 32.7231026, -68.2781372, 69.4640350
1: -28.6053123, 24.4596920, -33.1540871, 28.2778568, -56.8831711, 57.6137733
2: -38.1526184, 24.7707939, -44.2174034, 28.5451202, -66.6977386, 68.9881973
3: -40.3334618, 21.2773552, -46.7726402, 24.5852566, -64.9187164, 68.0499878
4: -37.8037033, 28.5302238, -43.6781502, 32.9869766, -70.7906723, 72.2083664
5: -34.0482864, 26.9581890, -39.4650764, 31.0813084, -65.1295853, 66.4232483
6: -32.4063263, 31.1801186, -37.3933792, 35.9815826, -68.3879013, 68.5734863
7: -34.6294899, 29.8969402, -40.0162201, 34.5093307, -69.1388092, 69.9131393
8: -41.6798592, 27.6523018, -48.2439919, 31.9987011, -73.6785583, 75.8962936
9: -32.0586395, 31.1621895, -36.9952660, 35.9585495, -68.0171890, 68.1574554

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7205029, upper bound: 106.7225731
time: 9.63 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7358983, upper bound: 106.7327698
time: 8.32 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -30.1891232, 24.0460186, -32.8912849, 26.2247200, -56.4138374, 56.9373016
1: -24.0445480, 20.6514492, -26.3511429, 22.5753899, -46.6199379, 47.0025902
2: -32.2060280, 20.9289417, -35.1938286, 22.7152443, -54.9212723, 56.1227646
3: -34.0082436, 17.9537144, -37.1199646, 19.6106071, -53.6188507, 55.0736771
4: -31.9257584, 24.1541386, -34.7933502, 26.3842850, -58.3100357, 58.9474869
5: -28.7468357, 22.8470268, -31.4890842, 24.9431286, -53.6899643, 54.3361130
6: -27.3880329, 26.4726028, -29.8149204, 28.8785706, -56.2666016, 56.2875099
7: -29.3281002, 25.3119297, -31.9067802, 27.5848331, -56.9129333, 57.2187119
8: -35.1237335, 23.3037052, -38.2919464, 25.4591770, -60.5829048, 61.5956497
9: -27.1509094, 26.3086071, -29.6109352, 28.6589451, -55.8098526, 55.9195404

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7214853, upper bound: 106.7159808
time: 11.22 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7209231, upper bound: 106.7158192
time: 10.62 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -33.9287224, 27.0298824, -40.2161140, 32.0669556, -65.9956818, 67.2459869
1: -27.1591053, 23.2587318, -32.4377708, 27.6869431, -54.8460426, 55.6965027
2: -36.2973976, 23.5752411, -43.2416191, 27.9450035, -64.2424011, 66.8168640
3: -38.4010010, 20.2501183, -45.7074623, 24.0982819, -62.4992790, 65.9575806
4: -35.9663963, 27.1698589, -42.7176552, 32.2895584, -68.2559509, 69.8875122
5: -32.3917694, 25.6757030, -38.6177139, 30.4522324, -62.8439980, 64.2934189
6: -30.8582458, 29.7365932, -36.5928535, 35.2487755, -66.1070251, 66.3294449
7: -32.9985924, 28.4730682, -39.1207161, 33.7391357, -66.7377319, 67.5937805
8: -39.6190834, 26.2788525, -47.2089272, 31.3984394, -71.0175247, 73.4877777
9: -30.5329399, 29.6359177, -36.2138100, 35.2288857, -65.7618179, 65.8497314

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7331166, upper bound: 106.7321800
time: 10.10 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319687
time: 8.96 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 20.43 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.43
Output dim: 0, lower bound: -106.7205639, upper bound: 106.7228340
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.43
Output dim: 0, lower bound: -106.7361426, upper bound: 106.7339479
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.43
Output dim: 0, lower bound: -106.7205029, upper bound: 106.7225731
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.43
Output dim: 0, lower bound: -106.7358983, upper bound: 106.7327698
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.43
Output dim: 0, lower bound: -106.7214853, upper bound: 106.7159808
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.43
Output dim: 0, lower bound: -106.7209231, upper bound: 106.7158192
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.43
Output dim: 0, lower bound: -106.7331166, upper bound: 106.7321800
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.43
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319687

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -26.9957714, 21.5279961, -35.3646851, 28.2104492, -55.2062225, 56.8926735
1: -21.4769115, 18.5021420, -28.4536438, 24.3343830, -45.8112907, 46.9557838
2: -28.8063183, 18.6599503, -37.9704590, 24.5304356, -53.3367538, 56.6304092
3: -30.2951717, 16.0209675, -40.0986443, 21.1298409, -51.4250107, 56.1196136
4: -28.5431118, 21.6506538, -37.5689201, 28.3675213, -56.9106331, 59.2195702
5: -25.7069683, 20.4726715, -33.9173622, 26.8446159, -52.5515823, 54.3900337
6: -24.4729385, 23.7288704, -32.1499901, 31.0261974, -55.4991379, 55.8788605
7: -26.2193050, 22.6755047, -34.3932381, 29.7284260, -55.9477310, 57.0687408
8: -31.3422241, 20.8133163, -41.3485680, 27.4296513, -58.7718735, 62.1618843
9: -24.3412437, 23.5257301, -31.8787251, 30.9115620, -55.2528076, 55.4044418

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7099831, upper bound: 106.7142188
time: 9.62 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7098455, upper bound: 106.7131605
time: 9.18 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -33.3200150, 26.5823078, -39.6654892, 31.6172333, -64.9372406, 66.2477951
1: -26.7578011, 22.9192848, -32.0245438, 27.3153019, -54.0731010, 54.9438248
2: -35.7314110, 23.1726189, -42.6806641, 27.5884781, -63.3198891, 65.8532639
3: -37.7270775, 19.9066143, -45.1123810, 23.7436314, -61.4707108, 65.0189896
4: -35.4014511, 26.7457314, -42.2029762, 31.8370152, -67.2384644, 68.9487000
5: -31.8808517, 25.2663097, -38.0951347, 30.0579700, -61.9388199, 63.3614311
6: -30.3533936, 29.2491856, -36.1102371, 34.7541695, -65.1075516, 65.3594131
7: -32.4427376, 28.0131454, -38.6414757, 33.3279648, -65.7707062, 66.6546097
8: -38.9699593, 25.8568001, -46.5749245, 30.9052410, -69.8751984, 72.4317093
9: -30.0644608, 29.1836205, -35.7313843, 34.7485466, -64.8130035, 64.9150085

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7211656, upper bound: 106.7190603
time: 10.39 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7208024, upper bound: 106.7177235
time: 9.21 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -24.0892925, 19.2044907, -34.8429337, 27.7701416, -51.8594322, 54.0474243
1: -19.0461349, 16.4932728, -27.9839668, 23.9428101, -42.9889412, 44.4772377
2: -25.6340561, 16.6217728, -37.3679924, 24.1092129, -49.7432709, 53.9897652
3: -26.9205036, 14.2887573, -39.5014153, 20.7840137, -47.7045174, 53.7901726
4: -25.4096088, 19.3177166, -36.9599419, 27.9540386, -53.3636360, 56.2776566
5: -22.8679161, 18.2798080, -33.3967285, 26.4065285, -49.2744446, 51.6765366
6: -21.8131008, 21.1807365, -31.6456013, 30.5662041, -52.3793030, 52.8263321
7: -23.3755836, 20.2423000, -33.8698120, 29.2671738, -52.6427574, 54.1121140
8: -27.8913364, 18.5256157, -40.6608810, 26.9660225, -54.8573532, 59.1864891
9: -21.7051945, 20.9425716, -31.3918495, 30.4026031, -52.1077957, 52.3344193

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7098457, upper bound: 106.7140513
time: 10.44 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7097096, upper bound: 106.7127839
time: 9.59 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -29.8976784, 23.8478050, -39.0317345, 31.0910835, -60.9887581, 62.8795395
1: -23.9091663, 20.5317116, -31.4471092, 26.8558865, -50.7650452, 51.9788208
2: -31.9769135, 20.7696762, -41.9686928, 27.0829449, -59.0598526, 62.7383690
3: -33.7357178, 17.8150349, -44.3800774, 23.3375015, -57.0732155, 62.1951141
4: -31.7052631, 23.9926414, -41.4635010, 31.3260708, -63.0313301, 65.4561462
5: -28.5436974, 22.6941681, -37.4648056, 29.5392265, -58.0829124, 60.1589737
6: -27.2051868, 26.2437420, -35.5026627, 34.2005119, -61.4057007, 61.7463989
7: -29.0928135, 25.1457291, -37.9859352, 32.7875252, -61.8803406, 63.1316643
8: -34.8609428, 23.1047668, -45.7419701, 30.3264904, -65.1874237, 68.8467331
9: -26.9664726, 26.1284008, -35.1478310, 34.1319275, -61.0983963, 61.2762222

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7210084, upper bound: 106.7181599
time: 10.62 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7206317, upper bound: 106.7165899
time: 10.08 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -20.1126461, 16.0194283, -29.5011196, 23.5208721, -43.6335144, 45.5205460
1: -15.6518440, 13.7178354, -23.5418453, 20.2162914, -35.8681297, 37.2596817
2: -21.2493458, 13.8934927, -31.4927998, 20.3536701, -41.6030159, 45.3862839
3: -22.3775387, 11.9015656, -33.1789818, 17.5287094, -39.9062462, 45.0805473
4: -21.1285667, 16.0514183, -31.1617870, 23.6505070, -44.7790756, 47.2132034
5: -18.9544697, 15.2378941, -28.1792030, 22.3918915, -41.3463593, 43.4170914
6: -18.2389011, 17.6378098, -26.7079620, 25.9065208, -44.1454201, 44.3457680
7: -19.4972973, 16.8888149, -28.6029015, 24.7474670, -44.2447662, 45.4917145
8: -23.1870499, 15.3893232, -34.2479324, 22.7460823, -45.9331245, 49.6372566
9: -18.0501575, 17.4099998, -26.5524902, 25.6646862, -43.7148438, 43.9624901

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7124142, upper bound: 106.7083499
time: 9.62 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7122656, upper bound: 106.7066360
time: 8.56 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -20.5351601, 16.3365402, -26.7706451, 21.3403530, -41.8755112, 43.1071777
1: -15.9746342, 13.9935007, -21.2551384, 18.3198051, -34.2944336, 35.2486343
2: -21.6943054, 14.1537418, -28.5072823, 18.4390278, -40.1333313, 42.6610222
3: -22.8717899, 12.1380672, -29.9918556, 15.8683491, -38.7401390, 42.1299210
4: -21.5554943, 16.3949165, -28.2162704, 21.4503345, -43.0058289, 44.6111870
5: -19.3479347, 15.5180016, -25.5131283, 20.3349190, -39.6828537, 41.0311241
6: -18.6202106, 18.0147762, -24.1983891, 23.5040894, -42.1242943, 42.2131653
7: -19.9093761, 17.2184010, -25.9282341, 22.4500256, -42.3594017, 43.1466370
8: -23.6797295, 15.6935968, -30.9797535, 20.5834827, -44.2632103, 46.6733475
9: -18.4146080, 17.7703285, -24.0691071, 23.2338333, -41.6484413, 41.8394356

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7116308, upper bound: 106.7080405
time: 9.35 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7115127, upper bound: 106.7064157
time: 9.61 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -22.9201889, 18.2385597, -36.4141846, 29.0323143, -51.9524956, 54.6527443
1: -18.0129681, 15.6695061, -29.2864952, 25.0455933, -43.0585632, 44.9560013
2: -24.3034382, 15.8932848, -39.0789642, 25.2695236, -49.5729599, 54.9722481
3: -25.6431770, 13.6194019, -41.2926331, 21.7853775, -47.4285507, 54.9120331
4: -24.1812000, 18.2996902, -38.6475906, 29.2134399, -53.3946342, 56.9472809
5: -21.6691113, 17.3736343, -34.9129791, 27.6055107, -49.2746201, 52.2866096
6: -20.8197842, 20.0805416, -33.1104927, 31.9296741, -52.7494545, 53.1910324
7: -22.2840919, 19.2735691, -35.3937988, 30.5721760, -52.8562622, 54.6673622
8: -26.5287151, 17.5605354, -42.6029205, 28.2946415, -54.8233566, 60.1634560
9: -20.5992908, 19.9207859, -32.7986603, 31.8436050, -52.4428825, 52.7194405

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7243811, upper bound: 106.7247474
time: 9.11 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7239363, upper bound: 106.7229655
time: 8.98 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -23.2609291, 18.4865112, -33.1956902, 26.4661980, -49.7271233, 51.6821976
1: -18.2581863, 15.8819284, -26.6132755, 22.8022003, -41.0603790, 42.4952049
2: -24.6476498, 16.0887527, -35.5451164, 23.0070190, -47.6546707, 51.6338692
3: -26.0274696, 13.8059540, -37.5401802, 19.8274899, -45.8549576, 51.3461342
4: -24.5050793, 18.5718517, -35.1749535, 26.6232643, -51.1283379, 53.7468033
5: -21.9763451, 17.5819817, -31.7809849, 25.1959648, -47.1723061, 49.3629684
6: -21.1096497, 20.3818245, -30.1598759, 29.1119728, -50.2216225, 50.5416946
7: -22.6100197, 19.5300560, -32.2521400, 27.8758163, -50.4858360, 51.7821960
8: -26.9089260, 17.8018475, -38.7172585, 25.6944828, -52.6034088, 56.5191040
9: -20.8871803, 20.2050629, -29.8933983, 28.9669590, -49.8541412, 50.0984573

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7231274, upper bound: 106.7244227
time: 9.09 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7227362, upper bound: 106.7227362
time: 8.48 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 18.97 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.97
Output dim: 0, lower bound: -106.7099831, upper bound: 106.7142188
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.97
Output dim: 0, lower bound: -106.7098455, upper bound: 106.7131605
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.97
Output dim: 0, lower bound: -106.7211656, upper bound: 106.7190603
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.97
Output dim: 0, lower bound: -106.7208024, upper bound: 106.7177235
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.97
Output dim: 0, lower bound: -106.7098457, upper bound: 106.7140513
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.97
Output dim: 0, lower bound: -106.7097096, upper bound: 106.7127839
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.97
Output dim: 0, lower bound: -106.7210084, upper bound: 106.7181599
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.97
Output dim: 0, lower bound: -106.7206317, upper bound: 106.7165899
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.97
Output dim: 0, lower bound: -106.7124142, upper bound: 106.7083499
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.97
Output dim: 0, lower bound: -106.7122656, upper bound: 106.7066360
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.97
Output dim: 0, lower bound: -106.7116308, upper bound: 106.7080405
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.97
Output dim: 0, lower bound: -106.7115127, upper bound: 106.7064157
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.97
Output dim: 0, lower bound: -106.7243811, upper bound: 106.7247474
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.97
Output dim: 0, lower bound: -106.7239363, upper bound: 106.7229655
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.97
Output dim: 0, lower bound: -106.7231274, upper bound: 106.7244227
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.97
Output dim: 0, lower bound: -106.7227362, upper bound: 106.7227362

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -25.4199505, 20.2968483, -28.9622650, 23.2001629, -48.6201057, 49.2591133
1: -20.1995220, 17.4381790, -23.2656231, 19.9802895, -40.1798096, 40.7038040
2: -27.1248112, 17.5865517, -31.0849590, 20.1585255, -47.2833366, 48.6715088
3: -28.5082645, 15.0946131, -32.7896004, 17.3009853, -45.8092499, 47.8842125
4: -26.8987598, 20.4150085, -30.8510494, 23.3263035, -50.2250633, 51.2660522
5: -24.1923218, 19.3153191, -27.7602749, 22.1553154, -46.3476295, 47.0755844
6: -23.0610428, 22.3635635, -26.3889866, 25.4674110, -48.5284500, 48.7525482
7: -24.7176323, 21.3948288, -28.2774162, 24.4914341, -49.2090607, 49.6722450
8: -29.5135212, 19.6060543, -33.8295097, 22.4497147, -51.9632339, 53.4355621
9: -22.9707584, 22.1816368, -26.2808723, 25.4192810, -48.3900375, 48.4625092

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7082090, upper bound: 106.7125491
time: 9.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7080054, upper bound: 106.7124384
time: 9.95 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -23.4622898, 18.7623463, -30.6349583, 24.5389099, -48.0011978, 49.3973045
1: -18.5980072, 16.1029072, -24.6815300, 21.1372433, -39.7352448, 40.7844391
2: -25.0181789, 16.2490101, -32.9209938, 21.3874092, -46.4055862, 49.1699982
3: -26.2810631, 13.9287529, -34.7520943, 18.3368969, -44.6179581, 48.6808434
4: -24.8229065, 18.8751869, -32.7152023, 24.6609325, -49.4838371, 51.5903854
5: -22.3056717, 17.8700085, -29.3817616, 23.4308739, -45.7365456, 47.2517700
6: -21.3214970, 20.6502190, -27.9872856, 26.9206238, -48.2421188, 48.6375008
7: -22.8379784, 19.7985992, -29.9916687, 25.9449234, -48.7828979, 49.7902679
8: -27.2327518, 18.0997982, -35.8534698, 23.8238525, -51.0566025, 53.9532623
9: -21.2570553, 20.4954262, -27.8316765, 27.0107441, -48.2677994, 48.3270950

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7080435, upper bound: 106.7113979
time: 9.01 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7077708, upper bound: 106.7112561
time: 11.45 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -27.6762810, 22.0559826, -24.2131805, 19.2533932, -46.9296646, 46.2691650
1: -22.0445633, 18.9762573, -19.1659737, 16.5571918, -38.6017532, 38.1422310
2: -29.5364552, 19.2347164, -25.7258968, 16.8020802, -46.3385353, 44.9606094
3: -31.1606102, 16.4955292, -27.1809673, 14.4567547, -45.6173630, 43.6764908
4: -29.3129807, 22.1844959, -25.5748425, 19.3458099, -48.6587906, 47.7593384
5: -26.3504639, 21.0114727, -23.0028076, 18.4415054, -44.7919693, 44.0142822
6: -25.1709576, 24.2679520, -22.0053139, 21.1791019, -46.3500557, 46.2732506
7: -26.9274330, 23.2755241, -23.5212917, 20.3737831, -47.3012161, 46.7968102
8: -32.2206688, 21.3390980, -28.0580463, 18.5030479, -50.7237167, 49.3971443
9: -24.9396725, 24.1664772, -21.7437706, 21.0284920, -45.9681625, 45.9102402

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7132700, upper bound: 106.7100578
time: 10.22 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7105392, upper bound: 106.7093552
time: 9.15 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -26.8485527, 21.3862514, -26.3563786, 20.9650383, -47.8135910, 47.7426300
1: -21.3512936, 18.4002438, -20.9579296, 18.0417442, -39.3930359, 39.3581696
2: -28.6298599, 18.6467533, -28.0852833, 18.2711067, -46.9009666, 46.7320251
3: -30.1944160, 15.9961681, -29.6797352, 15.7406979, -45.9351120, 45.6759033
4: -28.4243259, 21.5134983, -27.9009323, 21.0905266, -49.5148506, 49.4144287
5: -25.5322781, 20.3795700, -25.0730686, 20.0448265, -45.5771027, 45.4526367
6: -24.4077778, 23.5417728, -23.9610462, 23.0857639, -47.4935341, 47.5028152
7: -26.1290379, 22.5799580, -25.6603603, 22.1858044, -48.3148422, 48.2403183
8: -31.2273426, 20.6694393, -30.6203499, 20.1767044, -51.4040451, 51.2897873
9: -24.1918945, 23.4311275, -23.7092209, 22.9282608, -47.1201515, 47.1403465

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7128576, upper bound: 106.7089504
time: 9.90 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7101499, upper bound: 106.7082230
time: 8.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -22.6305408, 18.0681286, -28.4823265, 22.7907085, -45.4212494, 46.5504532
1: -17.8616238, 15.5115385, -22.8337727, 19.6164970, -37.4781151, 38.3453102
2: -24.0807800, 15.6275969, -30.5261478, 19.7656574, -43.8464355, 46.1537399
3: -25.2804756, 13.4275217, -32.2367249, 16.9806309, -42.2611084, 45.6642418
4: -23.8839989, 18.1808300, -30.2816238, 22.9482021, -46.8321953, 48.4624557
5: -21.4714851, 17.2063980, -27.2806129, 21.7457066, -43.2171898, 44.4870110
6: -20.5227566, 19.9105396, -25.9122105, 25.0481548, -45.5709114, 45.8227425
7: -21.9835854, 19.0565033, -27.7845688, 24.0639896, -46.0475693, 46.8410721
8: -26.2071705, 17.4144802, -33.1903229, 22.0136490, -48.2208176, 50.6048050
9: -20.4412174, 19.6993656, -25.8251133, 24.9496746, -45.3908844, 45.5244751

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7080416, upper bound: 106.7122333
time: 10.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7078746, upper bound: 106.7121821
time: 8.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -20.8287792, 16.6630268, -29.8888187, 23.9223995, -44.7511787, 46.5518456
1: -16.3743076, 14.2810850, -24.0221996, 20.5952606, -36.9695625, 38.3032837
2: -22.1331749, 14.3963461, -32.0818214, 20.8046932, -42.9378586, 46.4781647
3: -23.2316475, 12.3504953, -33.9015999, 17.8556156, -41.0872574, 46.2520905
4: -21.9704037, 16.7523270, -31.8669682, 24.0574226, -46.0278206, 48.6192932
5: -19.7352371, 15.8867941, -28.6394844, 22.8225708, -42.5578003, 44.5262794
6: -18.9114017, 18.3239212, -27.2808781, 26.2680664, -45.1794662, 45.6047974
7: -20.2493744, 17.5859280, -29.2430573, 25.3000317, -45.5494080, 46.8289871
8: -24.0951309, 16.0252323, -34.9096794, 23.1828156, -47.2779350, 50.9349098
9: -18.8665276, 18.1383820, -27.1476059, 26.3008385, -45.1673660, 45.2859840

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7078760, upper bound: 106.7109209
time: 9.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7076558, upper bound: 106.7108598
time: 8.96 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -24.7049465, 19.6785145, -24.1821976, 19.2078876, -43.9128342, 43.8607101
1: -19.5662022, 16.9207382, -19.0962524, 16.5188103, -36.0850067, 36.0169907
2: -26.2885647, 17.1498852, -25.6688042, 16.7365417, -43.0251083, 42.8186836
3: -27.7131271, 14.7177601, -27.1417999, 14.4112377, -42.1243668, 41.8595581
4: -26.1153755, 19.7913895, -25.4913960, 19.3307686, -45.4461403, 45.2827759
5: -23.4541054, 18.7685490, -22.9535522, 18.3779659, -41.8320656, 41.7220993
6: -22.4489651, 21.6604195, -21.9489517, 21.1543331, -43.6032982, 43.6093636
7: -24.0289860, 20.7874870, -23.4716568, 20.3165665, -44.3455505, 44.2591400
8: -28.6801853, 18.9807720, -27.9843216, 18.4638596, -47.1440430, 46.9650879
9: -22.2490215, 21.5322380, -21.6967735, 20.9726925, -43.2217140, 43.2290077

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7129620, upper bound: 106.7090198
time: 9.47 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7102897, upper bound: 106.7083145
time: 8.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -23.9339027, 19.0627651, -25.6319981, 20.3655682, -44.2994690, 44.6947594
1: -18.9196968, 16.3865185, -20.3196583, 17.5154705, -36.4351654, 36.7061729
2: -25.4448109, 16.6037769, -27.2632027, 17.7215366, -43.1663399, 43.8669815
3: -26.8162727, 14.2560320, -28.8204098, 15.2772751, -42.0935402, 43.0764351
4: -25.2835693, 19.1760254, -27.0671482, 20.5063000, -45.7898674, 46.2431717
5: -22.6965942, 18.1878834, -24.3450127, 19.4600830, -42.1566734, 42.5328979
6: -21.7449245, 20.9835949, -23.2691307, 22.4475002, -44.1924248, 44.2527237
7: -23.2801323, 20.1432381, -24.9240551, 21.5463200, -44.8264503, 45.0672913
8: -27.7625923, 18.3686695, -29.7253246, 19.5832386, -47.3458328, 48.0939941
9: -21.5539722, 20.8430729, -23.0257072, 22.2598877, -43.8138580, 43.8687820

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7126155, upper bound: 106.7076398
time: 10.21 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7098897, upper bound: 106.7069712
time: 9.45 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -18.8075047, 15.0088530, -23.6650734, 18.9625149, -37.7700195, 38.6739273
1: -14.5800047, 12.8323603, -18.7965221, 16.2648449, -30.8448486, 31.6288795
2: -19.8618279, 13.0068922, -25.2610950, 16.3804760, -36.2423019, 38.2679710
3: -20.9018784, 11.1269665, -26.5449467, 14.0629206, -34.9647942, 37.6719093
4: -19.7629299, 15.0287485, -25.0668964, 19.0588036, -38.8217278, 40.0956383
5: -17.7025928, 14.2782564, -22.5653572, 18.1010971, -35.8036880, 36.8436127
6: -17.0625248, 16.5056610, -21.4877472, 20.8362503, -37.8987732, 37.9934006
7: -18.2401180, 15.8249969, -23.0315990, 19.9861240, -38.2262421, 38.8565903
8: -21.6821022, 14.4072542, -27.4686718, 18.2667885, -39.9488869, 41.8759270
9: -16.9180813, 16.2837925, -21.4637337, 20.6823292, -37.6004105, 37.7475243

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7106326, upper bound: 106.7064994
time: 9.06 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7106188, upper bound: 106.7064985
time: 11.23 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -17.4854298, 13.9707317, -25.1708927, 20.1632290, -37.6486588, 39.1416245
1: -13.4781399, 11.9139013, -20.0825367, 17.2911072, -30.7692471, 31.9964371
2: -18.4306374, 12.1048613, -26.8974838, 17.4737148, -35.9043503, 39.0023384
3: -19.3837090, 10.3374834, -28.3205929, 14.9747286, -34.3584366, 38.6580696
4: -18.3505955, 13.9806976, -26.7338772, 20.2662029, -38.6167984, 40.7145691
5: -16.4279633, 13.2889347, -24.0267124, 19.2441349, -35.6720963, 37.3156471
6: -15.8587379, 15.3481302, -22.9111786, 22.1578960, -38.0166283, 38.2593040
7: -16.9451237, 14.7428160, -24.5448303, 21.3186855, -38.2638092, 39.2876472
8: -20.1367035, 13.4065838, -29.2463894, 19.4491482, -39.5858421, 42.6529655
9: -15.7521200, 15.1268806, -22.8770428, 22.0658970, -37.8180161, 38.0039215

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7103867, upper bound: 106.7047992
time: 9.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7103723, upper bound: 106.7048060
time: 10.37 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -19.2365551, 15.3298225, -21.2848225, 17.0581017, -36.2946548, 36.6146469
1: -14.9093456, 13.1173868, -16.7899303, 14.6225653, -29.5319099, 29.9073143
2: -20.3150253, 13.2715082, -22.6640396, 14.7114010, -35.0264282, 35.9355469
3: -21.4085693, 11.3678150, -23.7941170, 12.6392431, -34.0478134, 35.1619339
4: -20.1978836, 15.3765535, -22.4966602, 17.1539268, -37.3518066, 37.8732147
5: -18.1014957, 14.5632534, -20.2375927, 16.3011799, -34.4026718, 34.8008461
6: -17.4516754, 16.8906898, -19.3185196, 18.7366180, -36.1882935, 36.2092094
7: -18.6592026, 16.1602592, -20.6966209, 17.9891453, -36.6483421, 36.8568802
8: -22.1820450, 14.7171106, -24.6456165, 16.3955460, -38.5775909, 39.3627281
9: -17.2884407, 16.6511250, -19.3078785, 18.5537949, -35.8422356, 35.9590034

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7098313, upper bound: 106.7061463
time: 9.19 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7097525, upper bound: 106.7061288
time: 10.95 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -17.8769836, 14.2649250, -22.6602268, 18.1570415, -36.0340271, 36.9251518
1: -13.7793379, 12.1739941, -17.9773865, 15.5574570, -29.3367882, 30.1513786
2: -18.8469219, 12.3467331, -24.1589565, 15.7144728, -34.5613937, 36.5056915
3: -19.8478107, 10.5571976, -25.4127979, 13.4654160, -33.3132248, 35.9699898
4: -18.7476730, 14.2996168, -24.0263309, 18.2572155, -37.0048866, 38.3259430
5: -16.7899952, 13.5467644, -21.5742111, 17.3412895, -34.1312790, 35.1209641
6: -16.2171993, 15.7013159, -20.6237526, 19.9470730, -36.1642723, 36.3250618
7: -17.3318195, 15.0500727, -22.0828228, 19.2153091, -36.5471268, 37.1328926
8: -20.5928059, 13.6930161, -26.2780209, 17.4777145, -38.0705185, 39.9710388
9: -16.0964985, 15.4670992, -20.6040764, 19.8208408, -35.9173317, 36.0711708

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7097194, upper bound: 106.7045577
time: 8.98 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7096454, upper bound: 106.7045541
time: 10.21 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -21.5100422, 17.1417160, -29.9080944, 23.9410992, -45.4511414, 47.0498123
1: -16.8581791, 14.7133417, -24.0071259, 20.6171780, -37.4753571, 38.7204666
2: -22.7965069, 14.9266481, -32.0795593, 20.8243179, -43.6208267, 47.0062065
3: -24.0395317, 12.7774601, -33.8640518, 17.8919334, -41.9314651, 46.6415100
4: -22.6942787, 17.1962700, -31.8186245, 24.0783310, -46.7726021, 49.0148888
5: -20.3199749, 16.3346748, -28.6504269, 22.8411503, -43.1611252, 44.9850998
6: -19.5581894, 18.8499832, -27.2581539, 26.2778893, -45.8360786, 46.1081390
7: -20.9286156, 18.1177120, -29.1738548, 25.2440434, -46.1726608, 47.2915649
8: -24.8915920, 16.4993610, -34.9496078, 23.2214489, -48.1130409, 51.4489670
9: -19.3694229, 18.7099724, -27.1044083, 26.2631588, -45.6325798, 45.8143768

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7225730, upper bound: 106.7230400
time: 8.08 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7226043, upper bound: 106.7230443
time: 8.46 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -20.0336037, 15.9910660, -31.5487061, 25.2508469, -45.2844505, 47.5397682
1: -15.6350908, 13.6979818, -25.4001465, 21.7515049, -37.3865967, 39.0981216
2: -21.2092533, 13.9165831, -33.8787117, 22.0292053, -43.2384567, 47.7952881
3: -22.3431625, 11.8874636, -35.7836189, 18.9035435, -41.2467041, 47.6710815
4: -21.1229687, 16.0329857, -33.6468506, 25.3857117, -46.5086784, 49.6798363
5: -18.8975964, 15.2457848, -30.2375355, 24.0887489, -42.9863434, 45.4833183
6: -18.2237930, 17.5612621, -28.8161564, 27.7057972, -45.9295883, 46.3774185
7: -19.4915123, 16.9113560, -30.8560123, 26.6683617, -46.1598663, 47.7673645
8: -23.1702576, 15.3852215, -36.9240570, 24.5690193, -47.7392769, 52.3092804
9: -18.0768032, 17.4198914, -28.6275196, 27.8177910, -45.8945885, 46.0474091

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7221237, upper bound: 106.7211809
time: 9.22 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7221397, upper bound: 106.7211940
time: 8.35 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 18.94 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7082090, upper bound: 106.7125491
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7080054, upper bound: 106.7124384
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7080435, upper bound: 106.7113979
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7077708, upper bound: 106.7112561
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7132700, upper bound: 106.7100578
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7105392, upper bound: 106.7093552
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7128576, upper bound: 106.7089504
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7101499, upper bound: 106.7082230
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7080416, upper bound: 106.7122333
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7078746, upper bound: 106.7121821
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7078760, upper bound: 106.7109209
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7076558, upper bound: 106.7108598
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7129620, upper bound: 106.7090198
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7102897, upper bound: 106.7083145
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7126155, upper bound: 106.7076398
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7098897, upper bound: 106.7069712
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7106326, upper bound: 106.7064994
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7106188, upper bound: 106.7064985
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7103867, upper bound: 106.7047992
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7103723, upper bound: 106.7048060
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7098313, upper bound: 106.7061463
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7097525, upper bound: 106.7061288
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7097194, upper bound: 106.7045577
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7096454, upper bound: 106.7045541
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7225730, upper bound: 106.7230400
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7226043, upper bound: 106.7230443
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7221237, upper bound: 106.7211809
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.94
Output dim: 0, lower bound: -106.7221397, upper bound: 106.7211940
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.94
Output dim: 0, lower bound: -106.7231274, upper bound: 106.7244227
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.94
Output dim: 0, lower bound: -106.7227362, upper bound: 106.7227362
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=108.02017974853516
rel_dist={0: [-106.76706178863964, 106.76706178863964]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7489437, upper bound: 106.7478694
time: 10.98 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7476055, upper bound: 106.7476055
time: 9.09 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 20.22 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 20.22
Output dim: 0, lower bound: -106.7489437, upper bound: 106.7478694
IS_A2, status: Status.UNKNOWN, split count: 1, time: 20.22
Output dim: 0, lower bound: -106.7476055, upper bound: 106.7476055

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -43.2367897, 34.4949570, -48.3691521, 38.5639191, -81.8006973, 82.8641052
1: -34.9878044, 29.8007889, -39.2482681, 33.3601799, -68.3479767, 69.0490417
2: -46.5671349, 30.1880875, -52.1956902, 33.7793694, -80.3464966, 82.3837662
3: -49.2782364, 25.9414082, -55.2171669, 29.0664406, -78.3446808, 81.1585770
4: -46.0794792, 34.7557182, -51.5466881, 38.9383621, -85.0178375, 86.3024063
5: -41.5383759, 32.7077026, -46.5417137, 36.5443039, -78.0826645, 79.2494202
6: -39.4550514, 37.8932304, -44.1378059, 42.3557434, -81.8107910, 82.0310364
7: -42.1981354, 36.3016472, -47.2421303, 40.5871887, -82.7853241, 83.5437775
8: -51.0103607, 33.9446335, -57.2294846, 38.1155205, -89.1258850, 91.1741180
9: -38.9905930, 38.0316658, -43.6247635, 42.5535889, -81.5441818, 81.6564255

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7298387, upper bound: 106.7274888
time: 11.31 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7447645, upper bound: 106.7436956
time: 10.59 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -35.7654915, 28.4981251, -38.2508507, 30.4932709, -66.2587585, 66.7489700
1: -28.6854954, 24.5347595, -30.7832279, 26.2898273, -54.9753227, 55.3179703
2: -38.3050308, 24.8694439, -41.0445976, 26.6168213, -64.9218521, 65.9140244
3: -40.5462570, 21.3767605, -43.4179153, 22.9137669, -63.4600220, 64.7946777
4: -37.9453735, 28.6490612, -40.5940475, 30.6803169, -68.6256866, 69.2431030
5: -34.1815300, 27.0604343, -36.6654549, 28.9656162, -63.1471405, 63.7258911
6: -32.5468407, 31.3406620, -34.8163567, 33.4998932, -66.0467072, 66.1570206
7: -34.7955933, 30.0198460, -37.1915703, 32.1008759, -66.8964539, 67.2114105
8: -41.8391533, 27.7593937, -44.8514366, 29.7988777, -71.6380310, 72.6108322
9: -32.1872482, 31.2693481, -34.4225273, 33.4836082, -65.6708527, 65.6918640

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 240

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7291394, upper bound: 106.7272116
time: 11.44 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7434330, upper bound: 106.7434330
time: 9.43 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.17 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.17
Output dim: 0, lower bound: -106.7298387, upper bound: 106.7274888
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.17
Output dim: 0, lower bound: -106.7447645, upper bound: 106.7436956
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.17
Output dim: 0, lower bound: -106.7291394, upper bound: 106.7272116
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.17
Output dim: 0, lower bound: -106.7434330, upper bound: 106.7434330

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -32.4412804, 25.8810692, -34.1615524, 27.2521992, -59.6934814, 60.0426216
1: -26.0094242, 22.2894840, -27.4338417, 23.4837017, -49.4931221, 49.7233162
2: -34.7466660, 22.4985695, -36.6257324, 23.6393299, -58.3859940, 59.1243019
3: -36.6533623, 19.3535652, -38.6352501, 20.3914242, -57.0447845, 57.9888153
4: -34.3984909, 26.0442314, -36.2201729, 27.4257622, -61.8242493, 62.2644043
5: -31.0340519, 24.5953140, -32.7250404, 25.8979626, -56.9320107, 57.3203545
6: -29.4978848, 28.4933529, -31.0238972, 30.0114822, -59.5093651, 59.5172386
7: -31.5388927, 27.2480659, -33.1895142, 28.6745167, -60.2134094, 60.4375801
8: -37.8656883, 25.1579113, -39.8780632, 26.5209312, -64.3866196, 65.0359650
9: -29.2562752, 28.3600616, -30.7996044, 29.8390675, -59.0953445, 59.1596680

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7177953, upper bound: 106.7152618
time: 12.21 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7175585, upper bound: 106.7152064
time: 11.66 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -38.4892197, 30.7183990, -41.5321960, 33.1374092, -71.6266327, 72.2505951
1: -31.0535221, 26.5138493, -33.5750504, 28.6296597, -59.6831741, 60.0888939
2: -41.3764381, 26.8149529, -44.7160034, 28.9156189, -70.2920532, 71.5309601
3: -43.7532463, 23.0592537, -47.2765274, 24.9053574, -68.6585999, 70.3357697
4: -40.9692039, 30.9155254, -44.2149467, 33.3813477, -74.3505478, 75.1304703
5: -36.9176903, 29.1550827, -39.9067268, 31.4372654, -68.3549576, 69.0618057
6: -35.0924377, 33.7691879, -37.8485069, 36.4271355, -71.5195770, 71.6176910
7: -37.5092468, 32.3335152, -40.4806366, 34.8603668, -72.3696060, 72.8141479
8: -45.2269783, 30.0623169, -48.8890686, 32.5311050, -77.7580795, 78.9513779
9: -34.7182007, 33.8037491, -37.4485817, 36.4770927, -71.1952820, 71.2523346

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 240

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7337609, upper bound: 106.7322894
time: 11.84 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7334334, upper bound: 106.7322116
time: 11.71 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -26.4219093, 21.0288658, -26.1732311, 20.8484573, -47.2703629, 47.2020950
1: -20.8992443, 18.0448112, -20.7260227, 17.8897705, -38.7890167, 38.7708321
2: -28.1005287, 18.2565937, -27.8395329, 18.0442963, -46.1448135, 46.0961266
3: -29.6005211, 15.6756744, -29.2752571, 15.5157175, -45.1162376, 44.9509315
4: -27.8705673, 21.1188793, -27.5674057, 20.9542732, -48.8248291, 48.6862831
5: -25.0698833, 19.9848061, -24.8836174, 19.8411732, -44.9110565, 44.8684235
6: -23.9062233, 23.1886826, -23.6518478, 22.9746113, -46.8808289, 46.8405304
7: -25.6312981, 22.1409416, -25.3370781, 21.9241772, -47.5554733, 47.4780197
8: -30.6050854, 20.3250904, -30.2765408, 20.1332397, -50.7383270, 50.6016312
9: -23.7351170, 22.9625530, -23.5160713, 22.7200184, -46.4551315, 46.4786224

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7213286, upper bound: 106.7190300
time: 12.12 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7208061, upper bound: 106.7189006
time: 11.18 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -31.7124062, 25.2588253, -32.4111176, 25.8335495, -57.5459557, 57.6699448
1: -25.3131809, 21.7158470, -25.9418392, 22.2404480, -47.5536270, 47.6576843
2: -33.8765602, 22.0100956, -34.6613235, 22.4805698, -56.3571320, 56.6714172
3: -35.8089066, 18.8882370, -36.6115112, 19.3516369, -55.1605453, 55.4997482
4: -33.5788002, 25.3847256, -34.3125839, 25.9841652, -59.5629578, 59.6973076
5: -30.2292938, 23.9999199, -30.9832497, 24.5779247, -54.8072205, 54.9831619
6: -28.8097229, 27.7982845, -29.4558411, 28.4164810, -57.2262001, 57.2541275
7: -30.8308506, 26.6010342, -31.4790344, 27.1952610, -58.0261116, 58.0800667
8: -36.9531784, 24.5077076, -37.7853737, 25.0831299, -62.0363083, 62.2930756
9: -28.5323925, 27.6692600, -29.1743031, 28.2914696, -56.8238602, 56.8435593

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7324079, upper bound: 106.7320096
time: 12.46 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319330, upper bound: 106.7319330
time: 9.83 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.63 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.63
Output dim: 0, lower bound: -106.7177953, upper bound: 106.7152618
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.63
Output dim: 0, lower bound: -106.7175585, upper bound: 106.7152064
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.63
Output dim: 0, lower bound: -106.7337609, upper bound: 106.7322894
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.63
Output dim: 0, lower bound: -106.7334334, upper bound: 106.7322116
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.63
Output dim: 0, lower bound: -106.7213286, upper bound: 106.7190300
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.63
Output dim: 0, lower bound: -106.7208061, upper bound: 106.7189006
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.63
Output dim: 0, lower bound: -106.7324079, upper bound: 106.7320096
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.63
Output dim: 0, lower bound: -106.7319330, upper bound: 106.7319330

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -21.9112930, 17.4793530, -25.9194508, 20.6733532, -42.5846481, 43.3987999
1: -17.2616634, 15.0095425, -20.5982761, 17.7632179, -35.0248795, 35.6078186
2: -23.2830963, 15.1552277, -27.6469212, 17.8924370, -41.1755333, 42.8021469
3: -24.4625893, 12.9991989, -29.0532932, 15.3671722, -39.8297501, 42.0524902
4: -23.1155052, 17.5765362, -27.3941784, 20.7864475, -43.9019547, 44.9707146
5: -20.7752419, 16.6466198, -24.6812210, 19.6845341, -40.4597778, 41.3278389
6: -19.8997898, 19.2599487, -23.4833431, 22.7891178, -42.6889000, 42.7432899
7: -21.2933311, 18.4620266, -25.1678181, 21.7868938, -43.0802231, 43.6298447
8: -25.3606129, 16.8128548, -30.0546227, 19.9386806, -45.2992935, 46.8674774
9: -19.7661572, 19.0642414, -23.3697853, 22.5572109, -42.3233604, 42.4340172

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 109

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7081796, upper bound: 106.7064317
time: 11.38 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7081677, upper bound: 106.7058054
time: 12.59 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -21.3348961, 16.9982891, -23.4143219, 18.6637421, -39.9986382, 40.4126129
1: -16.7318630, 14.5804090, -18.4865780, 16.0247154, -32.7565765, 33.0669861
2: -22.6192570, 14.7028284, -24.8960438, 16.1266594, -38.7459106, 39.5988731
3: -23.7859974, 12.6247692, -26.1281796, 13.8773613, -37.6633530, 38.7529488
4: -22.4448662, 17.1067829, -24.6705532, 18.7779942, -41.2228622, 41.7773323
5: -20.1876183, 16.1733799, -22.2327404, 17.7915363, -37.9791565, 38.4061127
6: -19.3445339, 18.7445946, -21.1851463, 20.5851555, -39.9296875, 39.9297371
7: -20.7015400, 17.9378853, -22.6974010, 19.6764278, -40.3779640, 40.6352844
8: -24.6419392, 16.3373547, -27.0678310, 17.9744339, -42.6163712, 43.4051743
9: -19.2132683, 18.5197620, -21.0815315, 20.3131390, -39.5264053, 39.6012802

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7158111, upper bound: 106.7134523
time: 10.91 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7157963, upper bound: 106.7133922
time: 15.36 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -26.4805889, 21.1179295, -32.1583672, 25.6511364, -52.1317253, 53.2762985
1: -21.0874100, 18.1828022, -25.7996235, 22.1137409, -43.2011414, 43.9824219
2: -28.2756138, 18.4138317, -34.4670486, 22.3366146, -50.6122284, 52.8808823
3: -29.7970676, 15.7702799, -36.3846741, 19.1886139, -48.9856796, 52.1549530
4: -28.0925236, 21.2372437, -34.1517677, 25.8046150, -53.8971405, 55.3890076
5: -25.2058048, 20.1159077, -30.7676811, 24.4085312, -49.6143303, 50.8835831
6: -24.1083946, 23.2481499, -29.2707767, 28.2276878, -52.3360786, 52.5189285
7: -25.8049431, 22.3172855, -31.2995911, 27.0490761, -52.8540154, 53.6168747
8: -30.8274555, 20.3880043, -37.5506973, 24.8902111, -55.7176628, 57.9387016
9: -23.9055634, 23.1401920, -29.0134068, 28.1192398, -52.0248032, 52.1535988

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7243982, upper bound: 106.7236178
time: 10.82 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7243339, upper bound: 106.7229937
time: 10.61 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -25.7136154, 20.4890079, -29.1322212, 23.2400303, -48.9536438, 49.6212273
1: -20.4146061, 17.6278210, -23.2768574, 20.0004749, -40.4150772, 40.9046783
2: -27.4140682, 17.8362885, -31.1352196, 20.2073555, -47.6214218, 48.9715042
3: -28.9065361, 15.2872877, -32.8451080, 17.3474464, -46.2539825, 48.1323929
4: -27.2263603, 20.6234531, -30.8644505, 23.3788280, -50.6051865, 51.4879036
5: -24.4408951, 19.4971943, -27.8258781, 22.1427555, -46.5836487, 47.3230629
6: -23.3818588, 22.5771828, -26.4792099, 25.5696888, -48.9515457, 49.0563927
7: -25.0405293, 21.6547966, -28.3308258, 24.5097065, -49.5502357, 49.9856071
8: -29.8857784, 19.7516670, -33.9168549, 22.4682217, -52.3540001, 53.6685219
9: -23.1903267, 22.4422493, -26.2637348, 25.4148006, -48.6051178, 48.7059784

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7239687, upper bound: 106.7234755
time: 12.04 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7239234, upper bound: 106.7228989
time: 13.69 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -21.0348473, 16.8448772, -22.1780796, 17.7310581, -38.7659035, 39.0229568
1: -16.5111237, 14.4231853, -17.4778690, 15.2006550, -31.7117786, 31.9010544
2: -22.3513451, 14.5857134, -23.5830650, 15.3255529, -37.6768875, 38.1687775
3: -23.5250015, 12.4922609, -24.7701416, 13.1618376, -36.6868401, 37.2624016
4: -22.2291126, 16.9049454, -23.4013176, 17.8338051, -40.0629196, 40.3062630
5: -19.9079304, 16.0361004, -21.0443439, 16.9094658, -36.8173943, 37.0804443
6: -19.1126213, 18.4958763, -20.1007881, 19.5031033, -38.6157112, 38.5966568
7: -20.4814892, 17.7653637, -21.5295143, 18.6738548, -39.1553345, 39.2948761
8: -24.3671875, 16.2222118, -25.6584930, 17.0913906, -41.4585724, 41.8807068
9: -19.0608425, 18.3570957, -20.0515118, 19.3146763, -38.3755150, 38.4086075

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7195849, upper bound: 106.7173148
time: 11.46 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7195804, upper bound: 106.7172777
time: 10.98 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -22.6959553, 18.1611786, -20.6785564, 16.5531540, -39.2491074, 38.8397331
1: -17.9201069, 15.5488796, -16.2330933, 14.1669416, -32.0870438, 31.7819729
2: -24.1572742, 15.7722330, -21.9509735, 14.2977800, -38.4550514, 37.7232018
3: -25.4398956, 13.4808989, -23.0541897, 12.2721443, -37.7120399, 36.5350876
4: -24.0547714, 18.2381172, -21.7980556, 16.6354847, -40.6902542, 40.0361710
5: -21.5116940, 17.2891331, -19.5888977, 15.8020048, -37.3136978, 36.8780289
6: -20.6650295, 19.9494762, -18.7593021, 18.1797256, -38.8447456, 38.7087708
7: -22.1385803, 19.2109909, -20.0754585, 17.4539471, -39.5925179, 39.2864494
8: -26.3268356, 17.5194969, -23.8928871, 15.9251404, -42.2519684, 41.4123764
9: -20.6047916, 19.8611794, -18.7361755, 18.0017548, -38.6065445, 38.5973549

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7190648, upper bound: 106.7171919
time: 11.96 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7190591, upper bound: 106.7171519
time: 11.57 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.2542362, 16.9156322, -24.2996368, 19.3533726, -40.6076050, 41.2152672
1: -16.6130962, 14.5167589, -19.2071552, 16.6343803, -33.2474747, 33.7239113
2: -22.4941463, 14.7069502, -25.8325634, 16.8365631, -39.3307076, 40.5395126
3: -23.7131977, 12.6043634, -27.2066422, 14.4523754, -38.1655731, 39.8110046
4: -22.3751774, 16.9639473, -25.6547966, 19.4536591, -41.8288345, 42.6187363
5: -20.0581779, 16.1057568, -23.0617809, 18.4597244, -38.5179024, 39.1675377
6: -19.2942886, 18.6247482, -22.0511169, 21.3094444, -40.6037331, 40.6758652
7: -20.6396408, 17.8612614, -23.5959091, 20.4230843, -41.0627251, 41.4571571
8: -24.5470333, 16.2696819, -28.1465645, 18.6458645, -43.1928902, 44.4162445
9: -19.0910702, 18.4382477, -21.8619099, 21.1360493, -40.2271156, 40.3001556

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7305208, upper bound: 106.7300939
time: 10.93 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7305292, upper bound: 106.7301218
time: 11.29 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.6067619, 17.1779976, -22.5088749, 17.9205914, -39.5273514, 39.6868744
1: -16.8703098, 14.7386141, -17.6893616, 15.3930721, -32.2633820, 32.4279747
2: -22.8560448, 14.9164619, -23.8608189, 15.5665455, -38.4225883, 38.7772751
3: -24.1135826, 12.7966490, -25.1149979, 13.3819208, -37.4954948, 37.9116478
4: -22.7158833, 17.2475929, -23.6914577, 18.0207005, -40.7365837, 40.9390488
5: -20.3809090, 16.3248138, -21.3203812, 17.1121387, -37.4930458, 37.6451912
6: -19.6020050, 18.9402790, -20.4094448, 19.7261295, -39.3281288, 39.3497238
7: -20.9781761, 18.1272526, -21.8266392, 18.9088306, -39.8870087, 39.9538918
8: -24.9473629, 16.5203362, -25.9992409, 17.2436962, -42.1910591, 42.5195694
9: -19.3903160, 18.7340202, -20.2247143, 19.5237122, -38.9140282, 38.9587326

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7300079, upper bound: 106.7300106
time: 9.15 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7300467, upper bound: 106.7300467
time: 10.50 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.01 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.01
Output dim: 0, lower bound: -106.7081796, upper bound: 106.7064317
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.01
Output dim: 0, lower bound: -106.7081677, upper bound: 106.7058054
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.01
Output dim: 0, lower bound: -106.7158111, upper bound: 106.7134523
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.01
Output dim: 0, lower bound: -106.7157963, upper bound: 106.7133922
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.01
Output dim: 0, lower bound: -106.7243982, upper bound: 106.7236178
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.01
Output dim: 0, lower bound: -106.7243339, upper bound: 106.7229937
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.01
Output dim: 0, lower bound: -106.7239687, upper bound: 106.7234755
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.01
Output dim: 0, lower bound: -106.7239234, upper bound: 106.7228989
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.01
Output dim: 0, lower bound: -106.7195849, upper bound: 106.7173148
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.01
Output dim: 0, lower bound: -106.7195804, upper bound: 106.7172777
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.01
Output dim: 0, lower bound: -106.7190648, upper bound: 106.7171919
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.01
Output dim: 0, lower bound: -106.7190591, upper bound: 106.7171519
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.01
Output dim: 0, lower bound: -106.7305208, upper bound: 106.7300939
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.01
Output dim: 0, lower bound: -106.7305292, upper bound: 106.7301218
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.01
Output dim: 0, lower bound: -106.7300079, upper bound: 106.7300106
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.01
Output dim: 0, lower bound: -106.7300467, upper bound: 106.7300467

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -18.6095142, 14.9147387, -20.7179279, 16.6247292, -35.2342377, 35.6326637
1: -14.5433102, 12.7669582, -16.3647232, 14.2609768, -28.8042870, 29.1316814
2: -19.7520447, 12.8962793, -22.0948143, 14.3520603, -34.1040955, 34.9910889
3: -20.7266064, 11.0405989, -23.1957531, 12.3106670, -33.0372734, 34.2363510
4: -19.6470013, 14.9740934, -21.9539165, 16.7064877, -36.3534813, 36.9280090
5: -17.6065483, 14.2128830, -19.6979198, 15.8634005, -33.4699440, 33.9108047
6: -16.9230576, 16.3700867, -18.8582382, 18.2594051, -35.1824570, 35.2283211
7: -18.1186924, 15.7588291, -20.2058525, 17.5624180, -35.6811104, 35.9646797
8: -21.5357666, 14.3026409, -24.0446548, 15.9587765, -37.4945450, 38.3472862
9: -16.8916607, 16.2220097, -18.8554058, 18.1141815, -35.0058441, 35.0774040

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7063686, upper bound: 106.7045504
time: 10.87 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7063769, upper bound: 106.7045507
time: 12.97 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -17.0835819, 13.7080574, -21.9565067, 17.6140518, -34.6976318, 35.6645660
1: -13.2517309, 11.6866970, -17.4319115, 15.0983238, -28.3500557, 29.1186085
2: -18.0758762, 11.8530197, -23.4409351, 15.2676039, -33.3434792, 35.2939529
3: -18.9457245, 10.1256189, -24.6386986, 13.0491123, -31.9948349, 34.7643089
4: -17.9938679, 13.7513027, -23.3364105, 17.7030392, -35.6969070, 37.0877151
5: -16.1195431, 13.0620575, -20.9009857, 16.8056965, -32.9252357, 33.9630394
6: -15.5281591, 15.0168200, -20.0398750, 19.3432636, -34.8714218, 35.0566902
7: -16.6199226, 14.5003586, -21.4595013, 18.6762028, -35.2961273, 35.9598465
8: -19.7304649, 13.1400862, -25.5171013, 16.9419136, -36.6723747, 38.6571808
9: -15.5361691, 14.8713894, -20.0330143, 19.2556248, -34.7917938, 34.9044037

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7063442, upper bound: 106.7039178
time: 11.26 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7063488, upper bound: 106.7039193
time: 11.49 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -18.2969856, 14.5781174, -18.7116737, 14.9204025, -33.2173882, 33.2897873
1: -14.1752663, 12.4611435, -14.5424576, 12.7545958, -26.9298630, 27.0035973
2: -19.2939701, 12.5898743, -19.7433014, 12.8438377, -32.1378098, 32.3331718
3: -20.3030205, 10.7650375, -20.7383556, 11.0170784, -31.3200989, 31.5033913
4: -19.1505756, 14.6328621, -19.5753174, 14.9712772, -34.1218529, 34.2081795
5: -17.2374001, 13.8742704, -17.6736965, 14.2466993, -31.4841003, 31.5479641
6: -16.5675468, 16.0622787, -16.9070606, 16.4197445, -32.9872818, 32.9693375
7: -17.6930542, 15.3776188, -18.0505657, 15.7165737, -33.4096184, 33.4281845
8: -21.0459118, 13.9749575, -21.4933052, 14.3127041, -35.3586159, 35.4682579
9: -16.4354305, 15.8287992, -16.7970810, 16.1607990, -32.5962257, 32.6258774

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7068521, upper bound: 106.7041686
time: 13.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7060306, upper bound: 106.7039063
time: 10.39 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -18.8665161, 15.0286922, -19.9252071, 15.8785944, -34.7451057, 34.9538994
1: -14.6519680, 12.8526468, -15.5578394, 13.5822134, -28.2341805, 28.4104843
2: -19.9157887, 12.9867935, -21.0686474, 13.6951675, -33.6109467, 34.0554390
3: -20.9551067, 11.1006603, -22.1125431, 11.7297592, -32.6848602, 33.2131996
4: -19.7767086, 15.0978804, -20.8911591, 15.9541121, -35.7308197, 35.9890404
5: -17.7856255, 14.3015194, -18.8381863, 15.1513433, -32.9369698, 33.1397057
6: -17.0937557, 16.5611153, -18.0119705, 17.4909534, -34.5847092, 34.5730858
7: -18.2655029, 15.8549852, -19.2647495, 16.7415066, -35.0070114, 35.1197319
8: -21.7164783, 14.4148273, -22.9191322, 15.2534933, -36.9699631, 37.3339539
9: -16.9577255, 16.3376369, -17.9021816, 17.2383938, -34.1961136, 34.2398186

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7068254, upper bound: 106.7040870
time: 11.24 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7059942, upper bound: 106.7038051
time: 11.19 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -22.6739578, 18.1547546, -26.1506500, 20.9513512, -43.6253090, 44.3054047
1: -17.9977341, 15.6114721, -20.9378281, 18.0430584, -36.0407944, 36.5493011
2: -24.2027225, 15.8072624, -28.0267124, 18.2374496, -42.4401665, 43.8339767
3: -25.4787159, 13.5255089, -29.5495491, 15.6245661, -41.1032715, 43.0750580
4: -24.0847569, 18.2483635, -27.8617134, 21.0821857, -45.1669388, 46.1100731
5: -21.5680065, 17.3230877, -24.9933186, 20.0054436, -41.5734406, 42.3164062
6: -20.7096882, 19.9270248, -23.8731384, 23.0083084, -43.7179832, 43.8001633
7: -22.1680832, 19.2221012, -25.5772438, 22.1557274, -44.3237991, 44.7993431
8: -26.4065628, 17.4680805, -30.5321636, 20.2364674, -46.6430283, 48.0002441
9: -20.5888577, 19.8851070, -23.7729492, 22.9868088, -43.5756645, 43.6580582

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7226605, upper bound: 106.7218771
time: 11.25 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7226649, upper bound: 106.7218770
time: 11.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -20.7713680, 16.6450462, -27.4902153, 22.0230618, -42.7944298, 44.1352615
1: -16.4001503, 14.2917881, -22.0775623, 18.9708652, -35.3710098, 36.3693504
2: -22.1278725, 14.5005884, -29.5038605, 19.2336311, -41.3615036, 44.0044441
3: -23.2686157, 12.3657503, -31.1086540, 16.4602718, -39.7288857, 43.4744034
4: -22.0306988, 16.7405796, -29.3598919, 22.1478081, -44.1785049, 46.1004715
5: -19.7133675, 15.9103317, -26.2881622, 21.0296783, -40.7430458, 42.1984940
6: -18.9885273, 18.2444992, -25.1617870, 24.1800480, -43.1685753, 43.4062843
7: -20.3081436, 17.6555557, -26.9442978, 23.3419304, -43.6500702, 44.5998497
8: -24.1482754, 16.0136929, -32.1467743, 21.3428307, -45.4911041, 48.1604691
9: -18.9117413, 18.2035522, -25.0363426, 24.2720547, -43.1837959, 43.2398872

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7225847, upper bound: 106.7212338
time: 12.17 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7225834, upper bound: 106.7212373
time: 9.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -22.0699615, 17.6512947, -23.4895439, 18.8337975, -40.9037590, 41.1408348
1: -17.4397659, 15.1650133, -18.7003918, 16.1848202, -33.6245880, 33.8654060
2: -23.5064430, 15.3432751, -25.1020565, 16.3599205, -39.8663597, 40.4453239
3: -24.7626057, 13.1322889, -26.4405403, 14.0216360, -38.7842407, 39.5728264
4: -23.3819561, 17.7606716, -24.9615402, 18.9417896, -42.3237457, 42.7222099
5: -20.9551964, 16.8263340, -22.4074936, 18.0007267, -38.9559250, 39.2338257
6: -20.1255436, 19.3896313, -21.4327431, 20.6572990, -40.7828445, 40.8223724
7: -21.5508633, 18.6817226, -22.9558716, 19.9178524, -41.4687157, 41.6375923
8: -25.6535282, 16.9739609, -27.3633423, 18.1285172, -43.7820396, 44.3373032
9: -20.0168495, 19.3154297, -21.3472881, 20.6007347, -40.6175728, 40.6627121

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 255

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7222185, upper bound: 106.7217375
time: 11.46 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7222156, upper bound: 106.7217305
time: 12.28 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -20.1804790, 16.1596928, -24.5524063, 19.6890144, -39.8694916, 40.7120972
1: -15.8566151, 13.8515520, -19.6263046, 16.9133205, -32.7699356, 33.4778519
2: -21.4552479, 14.0477438, -26.2773876, 17.1556206, -38.6108704, 40.3251305
3: -22.5713997, 11.9784327, -27.7192078, 14.6649141, -37.2363129, 39.6976318
4: -21.3529358, 16.2589397, -26.1718292, 19.7952290, -41.1481628, 42.4307632
5: -19.1126633, 15.4175501, -23.4347687, 18.8134003, -37.9260635, 38.8523178
6: -18.4147663, 17.7246990, -22.4707794, 21.5967197, -40.0114784, 40.1954803
7: -19.7100983, 17.1234512, -24.0548286, 20.8951778, -40.6052704, 41.1782799
8: -23.4258156, 15.5401783, -28.6367455, 18.9729214, -42.3987350, 44.1769218
9: -18.3550968, 17.6542397, -22.3841000, 21.6144943, -39.9695892, 40.0383377

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7221768, upper bound: 106.7211345
time: 11.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7221730, upper bound: 106.7211356
time: 12.09 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -18.0870037, 14.5075321, -17.6636620, 14.1507092, -32.2377129, 32.1711960
1: -14.0307961, 12.3400822, -13.6802864, 12.0490923, -26.0798874, 26.0203686
2: -19.1091766, 12.5382404, -18.6222916, 12.1824303, -31.2916031, 31.1605301
3: -20.1084576, 10.6970329, -19.5796738, 10.4301434, -30.5386009, 30.2767067
4: -19.0058899, 14.5031347, -18.5000858, 14.1621571, -33.1680412, 33.0032158
5: -17.0459118, 13.7887440, -16.6663437, 13.4967604, -30.5426712, 30.4550877
6: -16.3820438, 15.8863926, -15.9612122, 15.5011349, -31.8831768, 31.8476048
7: -17.5563583, 15.2825937, -17.0601768, 14.8836823, -32.4400368, 32.3427696
8: -20.8624973, 13.9435949, -20.3011417, 13.5738907, -34.4363823, 34.2447357
9: -16.3561783, 15.7380123, -15.9327478, 15.3171053, -31.6732788, 31.6707554

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7021762, upper bound: 106.6994710
time: 11.89 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7010914, upper bound: 106.6990568
time: 13.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -18.6284351, 14.9357624, -18.8103275, 15.0554705, -33.6838989, 33.7460861
1: -14.4862452, 12.7156744, -14.6438885, 12.8373957, -27.3236332, 27.3595619
2: -19.6996422, 12.9171429, -19.8793983, 12.9872904, -32.6869316, 32.7965393
3: -20.7328129, 11.0165405, -20.8862686, 11.1069250, -31.8397369, 31.9028072
4: -19.6039371, 14.9466953, -19.7523651, 15.0935917, -34.6975250, 34.6990547
5: -17.5652142, 14.1980810, -17.7655411, 14.3577251, -31.9229393, 31.9636230
6: -16.8855743, 16.3607044, -17.0152988, 16.5131435, -33.3987198, 33.3760033
7: -18.1015053, 15.7389231, -18.2144985, 15.8542175, -33.9557114, 33.9534225
8: -21.5044079, 14.3602848, -21.6553555, 14.4581003, -35.9625015, 36.0156364
9: -16.8571873, 16.2239437, -16.9865971, 16.3388920, -33.1960793, 33.2105370

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 144

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7021024, upper bound: 106.6992225
time: 10.30 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7010558, upper bound: 106.6988785
time: 12.24 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.93 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7063686, upper bound: 106.7045504
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7063769, upper bound: 106.7045507
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7063442, upper bound: 106.7039178
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7063488, upper bound: 106.7039193
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7068521, upper bound: 106.7041686
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7060306, upper bound: 106.7039063
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7068254, upper bound: 106.7040870
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7059942, upper bound: 106.7038051
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7226605, upper bound: 106.7218771
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7226649, upper bound: 106.7218770
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7225847, upper bound: 106.7212338
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7225834, upper bound: 106.7212373
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7222185, upper bound: 106.7217375
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7222156, upper bound: 106.7217305
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7221768, upper bound: 106.7211345
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7221730, upper bound: 106.7211356
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7021762, upper bound: 106.6994710
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7010914, upper bound: 106.6990568
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7021024, upper bound: 106.6992225
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.93
Output dim: 0, lower bound: -106.7010558, upper bound: 106.6988785
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -106.7190648, upper bound: 106.7171919
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -106.7190591, upper bound: 106.7171519
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -106.7305208, upper bound: 106.7300939
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -106.7305292, upper bound: 106.7301218
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -106.7300079, upper bound: 106.7300106
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -106.7300467, upper bound: 106.7300467
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=108.02017974853516
rel_dist={0: [-106.76692036843552, 106.76692036843548]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1822.51 seconds
