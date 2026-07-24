## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 27.1733048946
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570)
1: (-16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204)
2: (-27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886)
3: (-24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909)
4: (-24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434)
5: (-18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721)
6: (-19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750)
7: (-22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555)
8: (-25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507)
9: (-17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555)

## BASE Result
execution time: IAR + LP analysis = 1.21 + 16.41 = 17.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -27.2006483, upper bound: 27.2006483


# Binary Search by BASE starts (time budget: 1982.38 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=36.862388610839844
rel_dist={2: [-27.200554238179546, 27.20055423068736]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=36.862388610839844
rel_dist={2: [-27.20050538908776, 27.200505386462353]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=36.862388610839844
rel_dist={2: [-27.200440708568877, 27.200440707169562]}

## Binary Search Result
Binary search time: 28.61 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1953.77 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1920535, upper bound: 27.1898814
time: 11.88 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1994759, upper bound: 27.1994759
time: 3.08 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.09 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.09
Output dim: 2, lower bound: -27.1920535, upper bound: 27.1898814
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.09
Output dim: 2, lower bound: -27.1994759, upper bound: 27.1994759

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -18.8142166, 14.2337074, -19.9730663, 15.1898537, -34.0040703, 34.2067719
1: -15.0117064, 12.6591454, -16.0186844, 13.5433626, -28.5550690, 28.6778297
2: -24.9902077, 8.0686264, -26.2949333, 9.1121101, -34.1023102, 34.3635597
3: -22.2631073, 10.0124874, -23.7272549, 10.8191662, -33.0822716, 33.7397423
4: -22.5668964, 13.6512918, -23.8643341, 14.5778427, -37.1447334, 37.5156250
5: -16.7592087, 14.4792290, -17.9284325, 15.3870077, -32.1462173, 32.4076614
6: -17.9284821, 15.2762775, -19.1249695, 16.2968063, -34.2252884, 34.4012451
7: -20.2259636, 14.6710501, -21.4438972, 15.7721729, -35.9981232, 36.1149368
8: -23.2887878, 13.2486000, -24.7556305, 14.1831455, -37.4719315, 38.0042305
9: -16.1759968, 19.0409184, -17.2642822, 20.1246185, -36.3006134, 36.3051949

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1885824, upper bound: 27.1867965
time: 7.12 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1898216, upper bound: 27.1875315
time: 6.67 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -20.0176010, 15.2209501, -20.7441998, 15.8020573, -35.8196564, 35.9651489
1: -16.0554142, 13.5701275, -16.6654358, 14.0918903, -30.1473045, 30.2355576
2: -26.3641472, 9.1117039, -27.2343006, 9.6280870, -35.9922333, 36.3460007
3: -23.7826290, 10.8370266, -24.6729698, 11.2894220, -35.0720520, 35.5099945
4: -23.9219360, 14.6047668, -24.7506828, 15.1801605, -39.1020927, 39.3554497
5: -17.9649963, 15.4206619, -18.6637077, 15.9828682, -33.9478569, 34.0843697
6: -19.1618958, 16.3336372, -19.8873863, 16.9607925, -36.1226883, 36.2210236
7: -21.4917870, 15.8007126, -22.2695312, 16.4308281, -37.9226112, 38.0702400
8: -24.8100815, 14.2101068, -25.7181702, 14.7784910, -39.5885658, 39.9282684
9: -17.2994194, 20.1760883, -17.9596100, 20.8685474, -38.1679611, 38.1356964

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1959498, upper bound: 27.1965690
time: 4.49 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1977583, upper bound: 27.1977583
time: 3.37 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 9.10 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 9.10
Output dim: 2, lower bound: -27.1885824, upper bound: 27.1867965
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 9.10
Output dim: 2, lower bound: -27.1898216, upper bound: 27.1875315
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 9.10
Output dim: 2, lower bound: -27.1959498, upper bound: 27.1965690
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 9.10
Output dim: 2, lower bound: -27.1977583, upper bound: 27.1977583

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -18.6881981, 14.1352386, -17.2603550, 13.0692110, -31.7574081, 31.3955936
1: -14.9099360, 12.5718298, -13.7848263, 11.6368256, -26.5467606, 26.3566551
2: -24.8352661, 7.9863062, -22.9789543, 7.3441563, -32.1794205, 30.9652576
3: -22.1152458, 9.9402122, -20.4592762, 9.2206459, -31.3358860, 30.3994884
4: -22.4215679, 13.5551701, -20.7190037, 12.5269394, -34.9485054, 34.2741737
5: -16.6416435, 14.3827877, -15.3795710, 13.3064041, -29.9480457, 29.7623558
6: -17.8031082, 15.1728334, -16.4207363, 14.0270367, -31.8301392, 31.5935707
7: -20.0904350, 14.5633793, -18.5514717, 13.4627638, -33.5531921, 33.1148529
8: -23.1302376, 13.1549406, -21.3504448, 12.1616306, -35.2918701, 34.5053864
9: -16.0630417, 18.9194870, -14.8357048, 17.4993057, -33.5623436, 33.7551880

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1885338, upper bound: 27.1866038
time: 5.01 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1885338, upper bound: 27.1867965
time: 5.09 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -18.7461510, 14.1804943, -18.6593094, 14.1545010, -32.9006500, 32.8398056
1: -14.9564085, 12.6118488, -14.9307041, 12.6028862, -27.5592937, 27.5425529
2: -24.9053440, 8.0253086, -24.7200794, 8.1791086, -33.0844498, 32.7453880
3: -22.1829300, 9.9738445, -22.1362953, 10.0119152, -32.1948395, 32.1101341
4: -22.4875774, 13.5994034, -22.3478928, 13.5572491, -36.0448227, 35.9472923
5: -16.6959190, 14.4268932, -16.6845589, 14.3740253, -31.0699368, 31.1114502
6: -17.8608284, 15.2202759, -17.8008595, 15.1893167, -33.0501442, 33.0211334
7: -20.1527557, 14.6134491, -20.0527859, 14.6284294, -34.7811852, 34.6662292
8: -23.2027855, 13.1980495, -23.1063347, 13.1766605, -36.3794479, 36.3043785
9: -16.1151943, 18.9745750, -16.0725269, 18.8723049, -34.9874992, 35.0471001

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1894515, upper bound: 27.1870641
time: 9.14 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1894515, upper bound: 27.1875315
time: 4.88 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -19.7946472, 15.0415154, -17.7370834, 13.4382677, -33.2329140, 32.7785988
1: -15.8703270, 13.4090233, -14.1783390, 11.9624004, -27.8327274, 27.5873566
2: -26.1010017, 8.9423523, -23.5818024, 7.6063967, -33.7073975, 32.5241470
3: -23.5116444, 10.6956062, -21.0418549, 9.4901848, -33.0018311, 31.7374611
4: -23.6664658, 14.4284973, -21.2769165, 12.8763294, -36.5427933, 35.7054138
5: -17.7495461, 15.2477484, -15.8263702, 13.6713247, -31.4208717, 31.0741196
6: -18.9341831, 16.1424179, -16.8846321, 14.4183712, -33.3525467, 33.0270462
7: -21.2533894, 15.6019115, -19.0637054, 13.8504381, -35.1038284, 34.6656189
8: -24.5298882, 14.0333500, -21.9471779, 12.5008354, -37.0307236, 35.9805298
9: -17.0945358, 19.9652634, -15.2502155, 17.9744301, -35.0689621, 35.2154770

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 58

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1954699, upper bound: 27.1954699
time: 4.20 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1954699, upper bound: 27.1965690
time: 5.28 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -19.9099045, 15.1349926, -19.2813759, 14.6328573, -34.5427628, 34.4163628
1: -15.9661188, 13.4929752, -15.4504633, 13.0406036, -29.0067215, 28.9434395
2: -26.2354603, 9.0329533, -25.4806252, 8.5678139, -34.8032684, 34.5135689
3: -23.6521759, 10.7698431, -22.9028149, 10.3804493, -34.0326233, 33.6726570
4: -23.7981071, 14.5194759, -23.0672989, 14.0238628, -37.8219681, 37.5867691
5: -17.8616161, 15.3373566, -17.2664795, 14.8502493, -32.7118607, 32.6038361
6: -19.0526199, 16.2416592, -18.4147415, 15.7105303, -34.7631493, 34.6563988
7: -21.3767357, 15.7061958, -20.7114697, 15.1501713, -36.5269089, 36.4176636
8: -24.6746674, 14.1256266, -23.8846569, 13.6342249, -38.3088913, 38.0102844
9: -17.2013168, 20.0735950, -16.6258526, 19.4761505, -36.6774635, 36.6994476

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1965690, upper bound: 27.1959498
time: 6.59 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1965690, upper bound: 27.1977583
time: 7.95 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 15.83 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.83
Output dim: 2, lower bound: -27.1885338, upper bound: 27.1866038
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.83
Output dim: 2, lower bound: -27.1885338, upper bound: 27.1867965
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.83
Output dim: 2, lower bound: -27.1894515, upper bound: 27.1870641
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.83
Output dim: 2, lower bound: -27.1894515, upper bound: 27.1875315
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.83
Output dim: 2, lower bound: -27.1954699, upper bound: 27.1954699
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.83
Output dim: 2, lower bound: -27.1954699, upper bound: 27.1965690
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.83
Output dim: 2, lower bound: -27.1965690, upper bound: 27.1959498
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.83
Output dim: 2, lower bound: -27.1965690, upper bound: 27.1977583

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -16.9130135, 12.7934694, -17.2603550, 13.0692110, -29.9822235, 30.0538216
1: -13.4790373, 11.3787155, -13.7848263, 11.6368256, -25.1158619, 25.1635380
2: -22.5727654, 7.0234098, -22.9789543, 7.3441563, -29.9169216, 30.0023594
3: -19.9994087, 8.9663801, -20.4592762, 9.2206459, -29.2200527, 29.4256554
4: -20.3578224, 12.2500992, -20.7190037, 12.5269394, -32.8847618, 32.9691010
5: -15.0268564, 13.0421124, -15.3795710, 13.3064041, -28.3332539, 28.4216805
6: -16.0804214, 13.7416916, -16.4207363, 14.0270367, -30.1074524, 30.1624279
7: -18.1732788, 13.1412086, -18.5514717, 13.4627638, -31.6360416, 31.6926804
8: -20.9308872, 11.8938808, -21.3504448, 12.1616306, -33.0925179, 33.2443237
9: -14.5212955, 17.1727448, -14.8357048, 17.4993057, -32.0205956, 32.0084457

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1683817, upper bound: 27.1677929
time: 6.86 seconds

## Relational analysis of IS_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1733220, upper bound: 27.1731773
time: 5.81 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1640973, upper bound: 27.1601617
time: 16.75 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -17.9587021, 13.5794115, -17.2603550, 13.0692110, -31.0279121, 30.8397636
1: -14.3199787, 12.0745945, -13.7848263, 11.6368256, -25.9568043, 25.8594208
2: -23.9166660, 7.5521708, -22.9789543, 7.3441563, -31.2608204, 30.5311241
3: -21.2553062, 9.5351715, -20.4592762, 9.2206459, -30.4759521, 29.9944477
4: -21.5772591, 13.0056839, -20.7190037, 12.5269394, -34.1041946, 33.7246857
5: -15.9738970, 13.8360071, -15.3795710, 13.3064041, -29.2802906, 29.2155762
6: -17.0821266, 14.5838366, -16.4207363, 14.0270367, -31.1091633, 31.0045738
7: -19.3057766, 13.9698009, -18.5514717, 13.4627638, -32.7685394, 32.5212708
8: -22.2208252, 12.6216097, -21.3504448, 12.1616306, -34.3824539, 33.9720497
9: -15.4244385, 18.2054062, -14.8357048, 17.4993057, -32.9237404, 33.0411110

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1683817, upper bound: 27.1677929
time: 5.75 seconds

## Relational analysis of IS_A1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1733220, upper bound: 27.1731924
time: 6.23 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1640973, upper bound: 27.1601617
time: 6.85 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -16.9130135, 12.7934694, -18.6593094, 14.1545010, -31.0675144, 31.4527683
1: -13.4790373, 11.3787155, -14.9307041, 12.6028862, -26.0819225, 26.3094158
2: -22.5727654, 7.0234098, -24.7200794, 8.1791086, -30.7518730, 31.7434883
3: -19.9994087, 8.9663801, -22.1362953, 10.0119152, -30.0113144, 31.1026764
4: -20.3578224, 12.2500992, -22.3478928, 13.5572491, -33.9150696, 34.5979881
5: -15.0268564, 13.0421124, -16.6845589, 14.3740253, -29.4008770, 29.7266693
6: -16.0804214, 13.7416916, -17.8008595, 15.1893167, -31.2697334, 31.5425510
7: -18.1732788, 13.1412086, -20.0527859, 14.6284294, -32.8017082, 33.1939926
8: -20.9308872, 11.8938808, -23.1063347, 13.1766605, -34.1075478, 35.0002060
9: -14.5212955, 17.1727448, -16.0725269, 18.8723049, -33.3936005, 33.2452621

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 217

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1845317, upper bound: 27.1847492
time: 15.83 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1845317, upper bound: 27.1870641
time: 6.33 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -17.9587021, 13.5794115, -18.6593094, 14.1545010, -32.1132050, 32.2387161
1: -14.3199787, 12.0745945, -14.9307041, 12.6028862, -26.9228649, 27.0052967
2: -23.9166660, 7.5521708, -24.7200794, 8.1791086, -32.0957718, 32.2722511
3: -21.2553062, 9.5351715, -22.1362953, 10.0119152, -31.2672195, 31.6714630
4: -21.5772591, 13.0056839, -22.3478928, 13.5572491, -35.1345024, 35.3535728
5: -15.9738970, 13.8360071, -16.6845589, 14.3740253, -30.3479137, 30.5205650
6: -17.0821266, 14.5838366, -17.8008595, 15.1893167, -32.2714424, 32.3846970
7: -19.3057766, 13.9698009, -20.0527859, 14.6284294, -33.9342041, 34.0225868
8: -22.2208252, 12.6216097, -23.1063347, 13.1766605, -35.3974838, 35.7279396
9: -15.4244385, 18.2054062, -16.0725269, 18.8723049, -34.2967453, 34.2779236

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1683817, upper bound: 27.1709547
time: 4.16 seconds

## Relational analysis of IS_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1884679, upper bound: 27.1865750
time: 4.77 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1885302, upper bound: 27.1874102
time: 4.20 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -17.3379898, 13.1264954, -17.7370834, 13.4382677, -30.7762566, 30.8635788
1: -13.8477888, 11.6870747, -14.1783390, 11.9624004, -25.8101883, 25.8654137
2: -23.0851631, 7.3696795, -23.5818024, 7.6063967, -30.6915569, 30.9514809
3: -20.5558205, 9.2606688, -21.0418549, 9.4901848, -30.0460052, 30.3025208
4: -20.8129292, 12.5806046, -21.2769165, 12.8763294, -33.6892586, 33.8575211
5: -15.4483376, 13.3657217, -15.8263702, 13.6713247, -29.1196632, 29.1920910
6: -16.4954891, 14.0895319, -16.8846321, 14.4183712, -30.9138565, 30.9741611
7: -18.6347389, 13.5205002, -19.0637054, 13.8504381, -32.4851761, 32.5842056
8: -21.4465790, 12.2142963, -21.9471779, 12.5008354, -33.9474144, 34.1614647
9: -14.9008160, 17.5803413, -15.2502155, 17.9744301, -32.8752441, 32.8305588

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1808845, upper bound: 27.1796893
time: 3.14 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1746993, upper bound: 27.1746993
time: 3.92 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -18.7389374, 14.2121658, -17.7370834, 13.4382677, -32.1772041, 31.9492493
1: -14.9956312, 12.6541805, -14.1783390, 11.9624004, -26.9580307, 26.8325195
2: -24.8272076, 8.2060719, -23.5818024, 7.6063967, -32.4336014, 31.7878742
3: -22.2329102, 10.0517826, -21.0418549, 9.4901848, -31.7230949, 31.0936375
4: -22.4428482, 13.6119614, -21.2769165, 12.8763294, -35.3191757, 34.8888779
5: -16.7553368, 14.4334555, -15.8263702, 13.6713247, -30.4266624, 30.2598267
6: -17.8754063, 15.2532148, -16.8846321, 14.4183712, -32.2937775, 32.1378479
7: -20.1374226, 14.6879034, -19.0637054, 13.8504381, -33.9878616, 33.7516098
8: -23.2031250, 13.2296448, -21.9471779, 12.5008354, -35.7039566, 35.1768150
9: -16.1392937, 18.9533577, -15.2502155, 17.9744301, -34.1137238, 34.2035675

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1796893, upper bound: 27.1815615
time: 4.71 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1746993, upper bound: 27.1772764
time: 12.52 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -17.3379898, 13.1264954, -19.2813759, 14.6328573, -31.9708481, 32.4078712
1: -13.8477888, 11.6870747, -15.4504633, 13.0406036, -26.8883934, 27.1375389
2: -23.0851631, 7.3696795, -25.4806252, 8.5678139, -31.6529713, 32.8502960
3: -20.5558205, 9.2606688, -22.9028149, 10.3804493, -30.9362698, 32.1634827
4: -20.8129292, 12.5806046, -23.0672989, 14.0238628, -34.8367920, 35.6478996
5: -15.4483376, 13.3657217, -17.2664795, 14.8502493, -30.2985840, 30.6322021
6: -16.4954891, 14.0895319, -18.4147415, 15.7105303, -32.2060165, 32.5042686
7: -18.6347389, 13.5205002, -20.7114697, 15.1501713, -33.7849121, 34.2319717
8: -21.4465790, 12.2142963, -23.8846569, 13.6342249, -35.0808029, 36.0989456
9: -14.9008160, 17.5803413, -16.6258526, 19.4761505, -34.3769608, 34.2061920

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1808845, upper bound: 27.1800856
time: 31.83 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1746993, upper bound: 27.1765785
time: 4.05 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -18.7389374, 14.2121658, -19.2813759, 14.6328573, -33.3717957, 33.4935417
1: -14.9956312, 12.6541805, -15.4504633, 13.0406036, -28.0362358, 28.1046448
2: -24.8272076, 8.2060719, -25.4806252, 8.5678139, -33.3950157, 33.6866913
3: -22.2329102, 10.0517826, -22.9028149, 10.3804493, -32.6133537, 32.9545975
4: -22.4428482, 13.6119614, -23.0672989, 14.0238628, -36.4667130, 36.6792603
5: -16.7553368, 14.4334555, -17.2664795, 14.8502493, -31.6055832, 31.6999359
6: -17.8754063, 15.2532148, -18.4147415, 15.7105303, -33.5859375, 33.6679573
7: -20.1374226, 14.6879034, -20.7114697, 15.1501713, -35.2875938, 35.3993721
8: -23.2031250, 13.2296448, -23.8846569, 13.6342249, -36.8373489, 37.1142960
9: -16.1392937, 18.9533577, -16.6258526, 19.4761505, -35.6154404, 35.5792084

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1808845, upper bound: 27.1859501
time: 11.25 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1746993, upper bound: 27.1821877
time: 13.82 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.36 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 26.36
Output dim: 2, lower bound: -27.1733220, upper bound: 27.1731773
IS_A1_B1_A1_A2, status: Status.VERIFIED, split count: 4, time: 26.36
Output dim: 2, lower bound: -27.1640973, upper bound: 27.1601617
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 26.36
Output dim: 2, lower bound: -27.1733220, upper bound: 27.1731924
IS_A1_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 26.36
Output dim: 2, lower bound: -27.1640973, upper bound: 27.1601617
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.36
Output dim: 2, lower bound: -27.1845317, upper bound: 27.1847492
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.36
Output dim: 2, lower bound: -27.1845317, upper bound: 27.1870641
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.36
Output dim: 2, lower bound: -27.1884679, upper bound: 27.1865750
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.36
Output dim: 2, lower bound: -27.1885302, upper bound: 27.1874102
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.36
Output dim: 2, lower bound: -27.1808845, upper bound: 27.1796893
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.36
Output dim: 2, lower bound: -27.1746993, upper bound: 27.1746993
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 26.36
Output dim: 2, lower bound: -27.1796893, upper bound: 27.1815615
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 26.36
Output dim: 2, lower bound: -27.1746993, upper bound: 27.1772764
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.36
Output dim: 2, lower bound: -27.1808845, upper bound: 27.1800856
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.36
Output dim: 2, lower bound: -27.1746993, upper bound: 27.1765785
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.36
Output dim: 2, lower bound: -27.1808845, upper bound: 27.1859501
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.36
Output dim: 2, lower bound: -27.1746993, upper bound: 27.1821877

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -16.7171860, 12.6482763, -17.2603550, 13.0692110, -29.7863960, 29.9086304
1: -13.3193378, 11.2485561, -13.7848263, 11.6368256, -24.9561634, 25.0333805
2: -22.3132019, 6.9381838, -22.9789543, 7.3441563, -29.6573582, 29.9171352
3: -19.7649956, 8.8653860, -20.4592762, 9.2206459, -28.9856396, 29.3246613
4: -20.1244507, 12.1102905, -20.7190037, 12.5269394, -32.6513901, 32.8292923
5: -14.8534393, 12.8930054, -15.3795710, 13.3064041, -28.1598434, 28.2725754
6: -15.8939276, 13.5840416, -16.4207363, 14.0270367, -29.9209633, 30.0047741
7: -17.9623451, 12.9902916, -18.5514717, 13.4627638, -31.4251041, 31.5417614
8: -20.6890526, 11.7587681, -21.3504448, 12.1616306, -32.8506851, 33.1092148
9: -14.3536777, 16.9764938, -14.8357048, 17.4993057, -31.8529816, 31.8121986

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1674100, upper bound: 27.1678391
time: 5.28 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1734211, upper bound: 27.1732203
time: 4.78 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -17.7524357, 13.4247513, -17.2603550, 13.0692110, -30.8216476, 30.6851063
1: -14.1526728, 11.9367161, -13.7848263, 11.6368256, -25.7894974, 25.7215424
2: -23.6464043, 7.4572730, -22.9789543, 7.3441563, -30.9905567, 30.4362278
3: -21.0095520, 9.4277763, -20.4592762, 9.2206459, -30.2301979, 29.8870525
4: -21.3332825, 12.8569775, -20.7190037, 12.5269394, -33.8602180, 33.5759735
5: -15.7884808, 13.6802244, -15.3795710, 13.3064041, -29.0948849, 29.0597916
6: -16.8835602, 14.4179802, -16.4207363, 14.0270367, -30.9105949, 30.8387127
7: -19.0850677, 13.8101225, -18.5514717, 13.4627638, -32.5478210, 32.3615952
8: -21.9640884, 12.4770546, -21.3504448, 12.1616306, -34.1257172, 33.8274956
9: -15.2475061, 17.9989738, -14.8357048, 17.4993057, -32.7468033, 32.8346786

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1672078, upper bound: 27.1677371
time: 5.78 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1733220, upper bound: 27.1731924
time: 15.11 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -16.9130135, 12.7934694, -17.8874741, 13.5268631, -30.4398708, 30.6809406
1: -13.4790373, 11.3787155, -14.2625885, 12.0275040, -25.5065384, 25.6412983
2: -22.5727654, 7.0234098, -23.8239956, 7.5214262, -30.0941906, 30.8474045
3: -19.9994087, 8.9663801, -21.1708260, 9.4986572, -29.4980583, 30.1372070
4: -20.3578224, 12.2500992, -21.4935303, 12.9550762, -33.3128967, 33.7436295
5: -15.0268564, 13.0421124, -15.9107056, 13.7824078, -28.8092613, 28.9528141
6: -16.0804214, 13.7416916, -17.0131073, 14.5267038, -30.6071205, 30.7547951
7: -18.1732788, 13.1412086, -19.2299156, 13.9158087, -32.0890884, 32.3711243
8: -20.9308872, 11.8938808, -22.1316624, 12.5717964, -33.5026817, 34.0255432
9: -14.5212955, 17.1727448, -15.3638668, 18.1342430, -32.6555328, 32.5366058

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1748327, upper bound: 27.1736494
time: 32.44 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1699012, upper bound: 27.1698450
time: 20.00 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -16.9130135, 12.7934694, -18.7389374, 14.2121658, -31.1251793, 31.5324020
1: -13.4790373, 11.3787155, -14.9956312, 12.6541805, -26.1332169, 26.3743439
2: -22.5727654, 7.0234098, -24.8272076, 8.2060719, -30.7788372, 31.8506165
3: -19.9994087, 8.9663801, -22.2329102, 10.0517826, -30.0511913, 31.1992912
4: -20.3578224, 12.2500992, -22.4428482, 13.6119614, -33.9697838, 34.6929474
5: -15.0268564, 13.0421124, -16.7553368, 14.4334555, -29.4603081, 29.7974491
6: -16.0804214, 13.7416916, -17.8754063, 15.2532148, -31.3336372, 31.6170959
7: -18.1732788, 13.1412086, -20.1374226, 14.6879034, -32.8611832, 33.2786331
8: -20.9308872, 11.8938808, -23.2031250, 13.2296448, -34.1605301, 35.0970078
9: -14.5212955, 17.1727448, -16.1392937, 18.9533577, -33.4746475, 33.3120308

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 217

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1846278, upper bound: 27.1869866
time: 5.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1848488, upper bound: 27.1870641
time: 3.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -17.8116379, 13.4689636, -16.8388138, 12.7800236, -30.5916615, 30.3077774
1: -14.2017250, 11.9769306, -13.4545059, 11.3784513, -25.5801735, 25.4314365
2: -23.7262001, 7.4808130, -22.3633842, 7.2924480, -31.0186424, 29.8441963
3: -21.0815697, 9.4567528, -19.9562359, 9.0386982, -30.1202679, 29.4129887
4: -21.4057541, 12.8988218, -20.1967354, 12.2421894, -33.6479416, 33.0955505
5: -15.8420420, 13.7254429, -15.0380020, 12.9931469, -28.8351841, 28.7634430
6: -16.9392128, 14.4666986, -16.0344925, 13.7113037, -30.6505127, 30.5011883
7: -19.1494064, 13.8543644, -18.1150646, 13.1914539, -32.3408585, 31.9694214
8: -22.0390587, 12.5171375, -20.8455811, 11.8811493, -33.9202080, 33.3627167
9: -15.2985401, 18.0590553, -14.5020943, 17.0527496, -32.3512802, 32.5611458

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 58

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1714541, upper bound: 27.1701669
time: 17.44 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1674540, upper bound: 27.1678335
time: 7.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1564681, upper bound: 27.1535056
time: 7.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -17.9587021, 13.5794115, -18.1657200, 13.7735100, -31.7322121, 31.7451267
1: -14.3199787, 12.0745945, -14.5284786, 12.2600584, -26.5800362, 26.6030731
2: -23.9166660, 7.5521708, -24.1014481, 7.8891907, -31.8058567, 31.6536179
3: -21.2553062, 9.5351715, -21.5406303, 9.7313061, -30.9866104, 31.0758018
4: -21.5772591, 13.0056839, -21.7749615, 13.1939297, -34.7711868, 34.7806473
5: -15.9738970, 13.8360071, -16.2300224, 13.9994030, -29.9732952, 30.0660248
6: -17.0821266, 14.5838366, -17.3106441, 14.7816753, -31.8638020, 31.8944817
7: -19.3057766, 13.9698009, -19.5328789, 14.2220621, -33.5278397, 33.5026779
8: -22.2208252, 12.6216097, -22.4897652, 12.8073092, -35.0281334, 35.1113739
9: -15.4244385, 18.2054062, -15.6377773, 18.3842278, -33.8086662, 33.8431854

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1719794, upper bound: 27.1709537
time: 12.92 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1738277, upper bound: 27.1735604
time: 5.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1640973, upper bound: 27.1601617
time: 13.07 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -17.3379898, 13.1264954, -17.5170212, 13.2687540, -30.6067410, 30.6435165
1: -13.8477888, 11.6870747, -13.9978123, 11.8115025, -25.6592903, 25.6848850
2: -23.0851631, 7.3696795, -23.3018532, 7.4864564, -30.5716190, 30.6715298
3: -20.5558205, 9.2606688, -20.7791500, 9.3680897, -29.9239101, 30.0398159
4: -20.8129292, 12.5806046, -21.0178070, 12.7147331, -33.5276527, 33.5984116
5: -15.4483376, 13.3657217, -15.6235085, 13.5027618, -28.9510937, 28.9892254
6: -16.4954891, 14.0895319, -16.6711426, 14.2380371, -30.7335186, 30.7606716
7: -18.6347389, 13.5205002, -18.8272114, 13.6730499, -32.3077888, 32.3477097
8: -21.4465790, 12.2142963, -21.6687336, 12.3438435, -33.7904205, 33.8830185
9: -14.9008160, 17.5803413, -15.0580196, 17.7569981, -32.6578140, 32.6383591

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1526034, upper bound: 27.1545854
time: 20.11 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1808656, upper bound: 27.1796677
time: 4.26 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1808830, upper bound: 27.1796893
time: 4.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -17.1812401, 13.0070133, -20.4667301, 15.4279709, -32.6092033, 33.4737434
1: -13.7191906, 11.5810089, -16.3545551, 13.7179642, -27.4371529, 27.9355640
2: -22.8850002, 7.2883635, -27.1952591, 8.5646553, -31.4496555, 34.4836197
3: -20.3659248, 9.1745129, -24.2848930, 10.8402805, -31.2062054, 33.4594040
4: -20.6283379, 12.4673033, -24.5229683, 14.7668667, -35.3951988, 36.9902725
5: -15.3040218, 13.2466192, -18.2149696, 15.7277212, -31.0317421, 31.4615841
6: -16.3466129, 13.9622383, -19.4812660, 16.5832462, -32.9298592, 33.4435043
7: -18.4663830, 13.3953695, -21.9660530, 15.8827620, -34.3491440, 35.3614197
8: -21.2525692, 12.1048193, -25.2900944, 14.3380995, -35.5906677, 37.3949127
9: -14.7652140, 17.4253101, -17.5430412, 20.7373371, -35.5025482, 34.9683456

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1743540, upper bound: 27.1745154
time: 3.75 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1746993, upper bound: 27.1746993
time: 13.63 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -18.4846287, 14.0152645, -17.7370834, 13.4382677, -31.9228935, 31.7523479
1: -14.7862778, 12.4748144, -14.1783390, 11.9624004, -26.7486782, 26.6531525
2: -24.5072727, 8.0530949, -23.5818024, 7.6063967, -32.1136665, 31.6348953
3: -21.9245892, 9.9061794, -21.0418549, 9.4901848, -31.4147739, 30.9480343
4: -22.1429405, 13.4248800, -21.2769165, 12.8763294, -35.0192719, 34.7017975
5: -16.5195637, 14.2372055, -15.8263702, 13.6713247, -30.1908875, 30.0635757
6: -17.6248817, 15.0410633, -16.8846321, 14.4183712, -32.0432510, 31.9256916
7: -19.8653374, 14.4775200, -19.0637054, 13.8504381, -33.7157745, 33.5412254
8: -22.8806992, 13.0422754, -21.9471779, 12.5008354, -35.3815346, 34.9894485
9: -15.9131203, 18.7040024, -15.2502155, 17.9744301, -33.8875504, 33.9542160

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1719695, upper bound: 27.1741968
time: 7.76 seconds

## Relational analysis of IS_A2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1709967, upper bound: 27.1733198
time: 10.03 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -21.9081173, 16.5225525, -17.5719223, 13.3106461, -35.2187653, 34.0944748
1: -17.5231323, 14.6790943, -14.0424738, 11.8486366, -29.3717651, 28.7215652
2: -29.0401115, 9.2706966, -23.3714809, 7.5147023, -36.5548134, 32.6421738
3: -26.0097790, 11.6112385, -20.8444939, 9.3981724, -35.4079514, 32.4557304
4: -26.2117062, 15.8026295, -21.0826111, 12.7544355, -38.9661407, 36.8852348
5: -19.5321407, 16.8187637, -15.6735744, 13.5447969, -33.0769386, 32.4923401
6: -20.8662186, 17.7550545, -16.7237473, 14.2827015, -35.1489182, 34.4787979
7: -23.5133858, 17.0352230, -18.8863697, 13.7169237, -37.2303009, 35.9215927
8: -27.0662708, 15.3426895, -21.7373886, 12.3822365, -39.4485054, 37.0800781
9: -18.7902985, 22.1685333, -15.1058178, 17.8109741, -36.6012650, 37.2743492

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1764215, upper bound: 27.1769292
time: 4.76 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1765785, upper bound: 27.1772757
time: 7.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -17.3379898, 13.1264954, -18.9890862, 14.4053783, -31.7433681, 32.1155815
1: -13.8477888, 11.6870747, -15.2084007, 12.8339100, -26.6816978, 26.8954716
2: -23.0851631, 7.3696795, -25.1194649, 8.3862009, -31.4713631, 32.4891357
3: -20.5558205, 9.2606688, -22.5460224, 10.2089367, -30.7647572, 31.8066845
4: -20.8129292, 12.5806046, -22.7240601, 13.8052349, -34.6181564, 35.3046646
5: -15.4483376, 13.3657217, -16.9932156, 14.6239452, -30.0722771, 30.3589344
6: -16.4954891, 14.0895319, -18.1241760, 15.4676208, -31.9631100, 32.2136955
7: -18.6347389, 13.5205002, -20.4010582, 14.9045773, -33.5393143, 33.9215584
8: -21.4465790, 12.2142963, -23.5148296, 13.4185171, -34.8650970, 35.7291183
9: -14.9008160, 17.5803413, -16.3637810, 19.1932983, -34.0941124, 33.9441223

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741968, upper bound: 27.1719695
time: 6.06 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1733198, upper bound: 27.1709967
time: 8.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 16.01 seconds
IS_A1_B1_A1_A1_B1, status: Status.VERIFIED, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1674100, upper bound: 27.1678391
IS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1734211, upper bound: 27.1732203
IS_A1_B1_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1672078, upper bound: 27.1677371
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1733220, upper bound: 27.1731924
IS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1748327, upper bound: 27.1736494
IS_A1_B2_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1699012, upper bound: 27.1698450
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1846278, upper bound: 27.1869866
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1848488, upper bound: 27.1870641
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1674540, upper bound: 27.1678335
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1564681, upper bound: 27.1535056
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1738277, upper bound: 27.1735604
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1640973, upper bound: 27.1601617
IS_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1808656, upper bound: 27.1796677
IS_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1808830, upper bound: 27.1796893
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1743540, upper bound: 27.1745154
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1746993, upper bound: 27.1746993
IS_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1719695, upper bound: 27.1741968
IS_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1709967, upper bound: 27.1733198
IS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1764215, upper bound: 27.1769292
IS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1765785, upper bound: 27.1772757
IS_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1741968, upper bound: 27.1719695
IS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 16.01
Output dim: 2, lower bound: -27.1733198, upper bound: 27.1709967
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.01
Output dim: 2, lower bound: -27.1746993, upper bound: 27.1765785
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.01
Output dim: 2, lower bound: -27.1808845, upper bound: 27.1859501
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.01
Output dim: 2, lower bound: -27.1746993, upper bound: 27.1821877
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=36.862388610839844
rel_dist={2: [-27.200554238179546, 27.20055423068736]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1972990, upper bound: 27.1969081
time: 5.64 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987716, upper bound: 27.1987716
time: 5.25 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.03 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.03
Output dim: 2, lower bound: -27.1972990, upper bound: 27.1969081
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.03
Output dim: 2, lower bound: -27.1987716, upper bound: 27.1987716

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -17.7370834, 13.4382677, -19.5840302, 14.8673496, -32.6044312, 33.0222969
1: -14.1783390, 11.9624004, -15.7014399, 13.2516689, -27.4300079, 27.6638412
2: -23.5818024, 7.6063967, -25.8618412, 8.7477245, -32.3295212, 33.4682388
3: -21.0418549, 9.4901848, -23.2685165, 10.5544968, -31.5963516, 32.7587013
4: -21.2769165, 12.8763294, -23.4271584, 14.2579031, -35.5348206, 36.3034897
5: -15.8263702, 13.6713247, -17.5454559, 15.0845242, -30.9108944, 31.2167816
6: -16.8846321, 14.4183712, -18.7129440, 15.9619741, -32.8466072, 33.1313171
7: -19.0637054, 13.8504381, -21.0339527, 15.3976727, -34.4613762, 34.8843880
8: -21.9471779, 12.5008354, -24.2660751, 13.8591270, -35.8062973, 36.7669067
9: -15.2502155, 17.9744301, -16.8939228, 19.7750587, -35.0252762, 34.8683472

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1966152, upper bound: 27.1966152
time: 8.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1966152, upper bound: 27.1969081
time: 4.50 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -19.2813759, 14.6328573, -20.0912285, 15.2757206, -34.5570946, 34.7240829
1: -15.4504633, 13.0406036, -16.1221123, 13.6216984, -29.0721626, 29.1627159
2: -25.4806252, 8.5678139, -26.4521561, 9.1441746, -34.6247940, 35.0199623
3: -22.9028149, 10.3804493, -23.8861313, 10.8818111, -33.7846260, 34.2665787
4: -23.0672989, 14.0238628, -24.0047455, 14.6549339, -37.7222328, 38.0286064
5: -17.2664795, 14.8502493, -18.0368500, 15.4769497, -32.7434311, 32.8871002
6: -18.4147415, 15.7105303, -19.2292881, 16.3985214, -34.8132591, 34.9398193
7: -20.7114697, 15.1501713, -21.5708580, 15.8566313, -36.5681000, 36.7210312
8: -23.8846569, 13.6342249, -24.8998566, 14.2604790, -38.1451340, 38.5340805
9: -16.6258526, 19.4761505, -17.3630238, 20.2495880, -36.8754425, 36.8391685

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1969081, upper bound: 27.1972990
time: 28.23 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1969081, upper bound: 27.1987716
time: 4.26 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 33.75 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 33.75
Output dim: 2, lower bound: -27.1966152, upper bound: 27.1966152
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 33.75
Output dim: 2, lower bound: -27.1966152, upper bound: 27.1969081
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 33.75
Output dim: 2, lower bound: -27.1969081, upper bound: 27.1972990
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 33.75
Output dim: 2, lower bound: -27.1969081, upper bound: 27.1987716

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -17.7370834, 13.4382677, -17.7370834, 13.4382677, -31.1753502, 31.1753502
1: -14.1783390, 11.9624004, -14.1783390, 11.9624004, -26.1407394, 26.1407394
2: -23.5818024, 7.6063967, -23.5818024, 7.6063967, -31.1881962, 31.1881981
3: -21.0418549, 9.4901848, -21.0418549, 9.4901848, -30.5320396, 30.5320396
4: -21.2769165, 12.8763294, -21.2769165, 12.8763294, -34.1532440, 34.1532440
5: -15.8263702, 13.6713247, -15.8263702, 13.6713247, -29.4976959, 29.4976959
6: -16.8846321, 14.4183712, -16.8846321, 14.4183712, -31.3030014, 31.3030033
7: -19.0637054, 13.8504381, -19.0637054, 13.8504381, -32.9141426, 32.9141426
8: -21.9471779, 12.5008354, -21.9471779, 12.5008354, -34.4480057, 34.4480057
9: -15.2502155, 17.9744301, -15.2502155, 17.9744301, -33.2246475, 33.2246475

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1866518, upper bound: 27.1856288
time: 6.12 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1953966, upper bound: 27.1953966
time: 4.77 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -17.7370834, 13.4382677, -19.2813759, 14.6328573, -32.3699417, 32.7196426
1: -14.1783390, 11.9624004, -15.4504633, 13.0406036, -27.2189426, 27.4128647
2: -23.5818024, 7.6063967, -25.4806252, 8.5678139, -32.1496048, 33.0870132
3: -21.0418549, 9.4901848, -22.9028149, 10.3804493, -31.4223042, 32.3929977
4: -21.2769165, 12.8763294, -23.0672989, 14.0238628, -35.3007812, 35.9436264
5: -15.8263702, 13.6713247, -17.2664795, 14.8502493, -30.6766167, 30.9378052
6: -16.8846321, 14.4183712, -18.4147415, 15.7105303, -32.5951614, 32.8331146
7: -19.0637054, 13.8504381, -20.7114697, 15.1501713, -34.2138748, 34.5619049
8: -21.9471779, 12.5008354, -23.8846569, 13.6342249, -35.5813980, 36.3854866
9: -15.2502155, 17.9744301, -16.6258526, 19.4761505, -34.7263641, 34.6002808

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1856288, upper bound: 27.1866688
time: 3.88 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1953966, upper bound: 27.1956661
time: 4.17 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -19.2813759, 14.6328573, -17.7370834, 13.4382677, -32.7196388, 32.3699417
1: -15.4504633, 13.0406036, -14.1783390, 11.9624004, -27.4128647, 27.2189426
2: -25.4806252, 8.5678139, -23.5818024, 7.6063967, -33.0870171, 32.1496124
3: -22.9028149, 10.3804493, -21.0418549, 9.4901848, -32.3929977, 31.4223042
4: -23.0672989, 14.0238628, -21.2769165, 12.8763294, -35.9436264, 35.3007812
5: -17.2664795, 14.8502493, -15.8263702, 13.6713247, -30.9378052, 30.6766186
6: -18.4147415, 15.7105303, -16.8846321, 14.4183712, -32.8331146, 32.5951614
7: -20.7114697, 15.1501713, -19.0637054, 13.8504381, -34.5619087, 34.2138748
8: -23.8846569, 13.6342249, -21.9471779, 12.5008354, -36.3854866, 35.5813980
9: -16.6258526, 19.4761505, -15.2502155, 17.9744301, -34.6002808, 34.7263641

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1866688, upper bound: 27.1857508
time: 5.05 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1956661, upper bound: 27.1960934
time: 4.85 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -19.2813759, 14.6328573, -19.2813759, 14.6328573, -33.9142342, 33.9142342
1: -15.4504633, 13.0406036, -15.4504633, 13.0406036, -28.4910660, 28.4910660
2: -25.4806252, 8.5678139, -25.4806252, 8.5678139, -34.0484314, 34.0484276
3: -22.9028149, 10.3804493, -22.9028149, 10.3804493, -33.2832565, 33.2832565
4: -23.0672989, 14.0238628, -23.0672989, 14.0238628, -37.0911636, 37.0911636
5: -17.2664795, 14.8502493, -17.2664795, 14.8502493, -32.1167297, 32.1167297
6: -18.4147415, 15.7105303, -18.4147415, 15.7105303, -34.1252670, 34.1252708
7: -20.7114697, 15.1501713, -20.7114697, 15.1501713, -35.8616409, 35.8616409
8: -23.8846569, 13.6342249, -23.8846569, 13.6342249, -37.5188828, 37.5188789
9: -16.6258526, 19.4761505, -16.6258526, 19.4761505, -36.1020050, 36.1020050

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1859429, upper bound: 27.1876655
time: 5.41 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1956661, upper bound: 27.1976968
time: 4.88 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 11.54 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 11.54
Output dim: 2, lower bound: -27.1866518, upper bound: 27.1856288
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 11.54
Output dim: 2, lower bound: -27.1953966, upper bound: 27.1953966
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 11.54
Output dim: 2, lower bound: -27.1856288, upper bound: 27.1866688
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 11.54
Output dim: 2, lower bound: -27.1953966, upper bound: 27.1956661
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 11.54
Output dim: 2, lower bound: -27.1866688, upper bound: 27.1857508
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 11.54
Output dim: 2, lower bound: -27.1956661, upper bound: 27.1960934
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 11.54
Output dim: 2, lower bound: -27.1859429, upper bound: 27.1876655
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 11.54
Output dim: 2, lower bound: -27.1956661, upper bound: 27.1976968

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -16.8317299, 12.7335157, -16.6347942, 12.5983982, -29.4301281, 29.3683090
1: -13.4129200, 11.3248215, -13.2749472, 11.2176657, -24.6305809, 24.5997696
2: -22.4660797, 6.9903717, -22.1815758, 7.0285330, -29.4946136, 29.1719475
3: -19.9012680, 8.9243526, -19.7020493, 8.8759098, -28.7771759, 28.6263962
4: -20.2617760, 12.1930265, -19.9911423, 12.0760460, -32.3378220, 32.1841660
5: -14.9546680, 12.9803066, -14.8098221, 12.8325138, -27.7871799, 27.7901249
6: -16.0028496, 13.6762362, -15.8234844, 13.5238495, -29.5266991, 29.4997158
7: -18.0861378, 13.0791740, -17.8823700, 12.9650517, -31.0511875, 30.9615440
8: -20.8303032, 11.8382320, -20.5825214, 11.7246342, -32.5549393, 32.4207535
9: -14.4516153, 17.0915165, -14.2965860, 16.8806686, -31.3322830, 31.3881035

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1844891, upper bound: 27.1844891
time: 13.32 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1844891, upper bound: 27.1856288
time: 6.33 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -17.3379898, 13.1264954, -17.7370834, 13.4382677, -30.7762566, 30.8635788
1: -13.8477888, 11.6870747, -14.1783390, 11.9624004, -25.8101883, 25.8654137
2: -23.0851631, 7.3696795, -23.5818024, 7.6063967, -30.6915569, 30.9514809
3: -20.5558205, 9.2606688, -21.0418549, 9.4901848, -30.0460052, 30.3025208
4: -20.8129292, 12.5806046, -21.2769165, 12.8763294, -33.6892586, 33.8575211
5: -15.4483376, 13.3657217, -15.8263702, 13.6713247, -29.1196632, 29.1920910
6: -16.4954891, 14.0895319, -16.8846321, 14.4183712, -30.9138565, 30.9741611
7: -18.6347389, 13.5205002, -19.0637054, 13.8504381, -32.4851761, 32.5842056
8: -21.4465790, 12.2142963, -21.9471779, 12.5008354, -33.9474144, 34.1614647
9: -14.9008160, 17.5803413, -15.2502155, 17.9744301, -32.8752441, 32.8305588

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1856288, upper bound: 27.1866518
time: 4.93 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1856288, upper bound: 27.1953966
time: 5.71 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -16.6347942, 12.5983982, -17.8874741, 13.5268631, -30.1616573, 30.4858723
1: -13.2749472, 11.2176657, -14.2625885, 12.0275040, -25.3024521, 25.4802494
2: -22.1815758, 7.0285330, -23.8239956, 7.5214262, -29.7030029, 30.8525257
3: -19.7020493, 8.8759098, -21.1708260, 9.4986572, -29.2006989, 30.0467358
4: -19.9911423, 12.0760460, -21.4935303, 12.9550762, -32.9462166, 33.5695763
5: -14.8098221, 12.8325138, -15.9107056, 13.7824078, -28.5922279, 28.7432117
6: -15.8234844, 13.5238495, -17.0131073, 14.5267038, -30.3501854, 30.5369568
7: -17.8823700, 12.9650517, -19.2299156, 13.9158087, -31.7981796, 32.1949615
8: -20.5825214, 11.7246342, -22.1316624, 12.5717964, -33.1543159, 33.8562927
9: -14.2965860, 16.8806686, -15.3638668, 18.1342430, -32.4308243, 32.2445374

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1847268, upper bound: 27.1846315
time: 5.15 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1847268, upper bound: 27.1866688
time: 4.60 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -17.7370834, 13.4382677, -18.7389374, 14.2121658, -31.9492493, 32.1772041
1: -14.1783390, 11.9624004, -14.9956312, 12.6541805, -26.8325195, 26.9580307
2: -23.5818024, 7.6063967, -24.8272076, 8.2060719, -31.7878704, 32.4336014
3: -21.0418549, 9.4901848, -22.2329102, 10.0517826, -31.0936375, 31.7230949
4: -21.2769165, 12.8763294, -22.4428482, 13.6119614, -34.8888779, 35.3191757
5: -15.8263702, 13.6713247, -16.7553368, 14.4334555, -30.2598267, 30.4266624
6: -16.8846321, 14.4183712, -17.8754063, 15.2532148, -32.1378479, 32.2937775
7: -19.0637054, 13.8504381, -20.1374226, 14.6879034, -33.7516098, 33.9878616
8: -21.9471779, 12.5008354, -23.2031250, 13.2296448, -35.1768150, 35.7039566
9: -15.2502155, 17.9744301, -16.1392937, 18.9533577, -34.2035751, 34.1137199

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1871831, upper bound: 27.1859430
time: 7.00 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1871831, upper bound: 27.1956661
time: 3.73 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -17.8874741, 13.5268631, -16.6347942, 12.5983982, -30.4858723, 30.1616573
1: -14.2625885, 12.0275040, -13.2749472, 11.2176657, -25.4802513, 25.3024521
2: -23.8239956, 7.5214262, -22.1815758, 7.0285330, -30.8525276, 29.7030010
3: -21.1708260, 9.4986572, -19.7020493, 8.8759098, -30.0467358, 29.2007027
4: -21.4935303, 12.9550762, -19.9911423, 12.0760460, -33.5695763, 32.9462204
5: -15.9107056, 13.7824078, -14.8098221, 12.8325138, -28.7432117, 28.5922260
6: -17.0131073, 14.5267038, -15.8234844, 13.5238495, -30.5369568, 30.3501835
7: -19.2299156, 13.9158087, -17.8823700, 12.9650517, -32.1949615, 31.7981796
8: -22.1316624, 12.5717964, -20.5825214, 11.7246342, -33.8562965, 33.1543159
9: -15.3638668, 18.1342430, -14.2965860, 16.8806686, -32.2445374, 32.4308281

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1846315, upper bound: 27.1847268
time: 3.27 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1846315, upper bound: 27.1857508
time: 3.84 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -18.7389374, 14.2121658, -17.7370834, 13.4382677, -32.1772041, 31.9492493
1: -14.9956312, 12.6541805, -14.1783390, 11.9624004, -26.9580307, 26.8325195
2: -24.8272076, 8.2060719, -23.5818024, 7.6063967, -32.4336014, 31.7878742
3: -22.2329102, 10.0517826, -21.0418549, 9.4901848, -31.7230949, 31.0936375
4: -22.4428482, 13.6119614, -21.2769165, 12.8763294, -35.3191757, 34.8888779
5: -16.7553368, 14.4334555, -15.8263702, 13.6713247, -30.4266624, 30.2598267
6: -17.8754063, 15.2532148, -16.8846321, 14.4183712, -32.2937775, 32.1378479
7: -20.1374226, 14.6879034, -19.0637054, 13.8504381, -33.9878616, 33.7516098
8: -23.2031250, 13.2296448, -21.9471779, 12.5008354, -35.7039566, 35.1768150
9: -16.1392937, 18.9533577, -15.2502155, 17.9744301, -34.1137238, 34.2035675

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1859429, upper bound: 27.1871831
time: 6.78 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1859429, upper bound: 27.1960934
time: 5.19 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -17.9179478, 13.5797911, -17.8874741, 13.5268631, -31.4447994, 31.4672661
1: -14.3123016, 12.0825882, -14.2625885, 12.0275040, -26.3398056, 26.3451710
2: -23.7918434, 7.7346840, -23.8239956, 7.5214262, -31.3132706, 31.5586796
3: -21.2268486, 9.5862627, -21.1708260, 9.4986572, -30.7254982, 30.7570877
4: -21.4794617, 13.0140543, -21.4935303, 12.9550762, -34.4345360, 34.5075836
5: -15.9881611, 13.8063536, -15.9107056, 13.7824078, -29.7705688, 29.7170582
6: -17.0717278, 14.5661659, -17.0131073, 14.5267038, -31.5984306, 31.5792732
7: -19.2609653, 14.0104275, -19.2299156, 13.9158087, -33.1767731, 33.2403336
8: -22.1756287, 12.6353064, -22.1316624, 12.5717964, -34.7474213, 34.7669678
9: -15.4163799, 18.1387329, -15.3638668, 18.1342430, -33.5506210, 33.5026016

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1852035, upper bound: 27.1852029
time: 3.89 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1852035, upper bound: 27.1876654
time: 5.14 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -19.2813759, 14.6328573, -18.7389374, 14.2121658, -33.4935417, 33.3717957
1: -15.4504633, 13.0406036, -14.9956312, 12.6541805, -28.1046448, 28.0362358
2: -25.4806252, 8.5678139, -24.8272076, 8.2060719, -33.6866875, 33.3950195
3: -22.9028149, 10.3804493, -22.2329102, 10.0517826, -32.9545975, 32.6133537
4: -23.0672989, 14.0238628, -22.4428482, 13.6119614, -36.6792603, 36.4667130
5: -17.2664795, 14.8502493, -16.7553368, 14.4334555, -31.6999359, 31.6055870
6: -18.4147415, 15.7105303, -17.8754063, 15.2532148, -33.6679573, 33.5859375
7: -20.7114697, 15.1501713, -20.1374226, 14.6879034, -35.3993721, 35.2875938
8: -23.8846569, 13.6342249, -23.2031250, 13.2296448, -37.1142921, 36.8373489
9: -16.6258526, 19.4761505, -16.1392937, 18.9533577, -35.5792084, 35.6154366

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1874860, upper bound: 27.1864186
time: 7.22 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1874860, upper bound: 27.1976968
time: 6.44 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 14.93 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.93
Output dim: 2, lower bound: -27.1844891, upper bound: 27.1844891
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.93
Output dim: 2, lower bound: -27.1844891, upper bound: 27.1856288
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.93
Output dim: 2, lower bound: -27.1856288, upper bound: 27.1866518
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.93
Output dim: 2, lower bound: -27.1856288, upper bound: 27.1953966
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 14.93
Output dim: 2, lower bound: -27.1847268, upper bound: 27.1846315
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 14.93
Output dim: 2, lower bound: -27.1847268, upper bound: 27.1866688
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 14.93
Output dim: 2, lower bound: -27.1871831, upper bound: 27.1859430
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 14.93
Output dim: 2, lower bound: -27.1871831, upper bound: 27.1956661
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.93
Output dim: 2, lower bound: -27.1846315, upper bound: 27.1847268
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.93
Output dim: 2, lower bound: -27.1846315, upper bound: 27.1857508
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.93
Output dim: 2, lower bound: -27.1859429, upper bound: 27.1871831
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.93
Output dim: 2, lower bound: -27.1859429, upper bound: 27.1960934
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 14.93
Output dim: 2, lower bound: -27.1852035, upper bound: 27.1852029
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 14.93
Output dim: 2, lower bound: -27.1852035, upper bound: 27.1876654
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 14.93
Output dim: 2, lower bound: -27.1874860, upper bound: 27.1864186
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 14.93
Output dim: 2, lower bound: -27.1874860, upper bound: 27.1976968

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -16.8317299, 12.7335157, -16.8317299, 12.7335157, -29.5652428, 29.5652428
1: -13.4129200, 11.3248215, -13.4129200, 11.3248215, -24.7377396, 24.7377415
2: -22.4660797, 6.9903717, -22.4660797, 6.9903717, -29.4564514, 29.4564514
3: -19.9012680, 8.9243526, -19.9012680, 8.9243526, -28.8256187, 28.8256149
4: -20.2617760, 12.1930265, -20.2617760, 12.1930265, -32.4548035, 32.4548035
5: -14.9546680, 12.9803066, -14.9546680, 12.9803066, -27.9349747, 27.9349747
6: -16.0028496, 13.6762362, -16.0028496, 13.6762362, -29.6790848, 29.6790810
7: -18.0861378, 13.0791740, -18.0861378, 13.0791740, -31.1652985, 31.1653023
8: -20.8303032, 11.8382320, -20.8303032, 11.8382320, -32.6685333, 32.6685333
9: -14.4516153, 17.0915165, -14.4516153, 17.0915165, -31.5431328, 31.5431328

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1723257, upper bound: 27.1717367
time: 3.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1697128, upper bound: 27.1697128
time: 16.27 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -16.8317299, 12.7335157, -17.3238983, 13.1115246, -29.9432545, 30.0574112
1: -13.4129200, 11.3248215, -13.8358364, 11.6765242, -25.0894432, 25.1606579
2: -22.4660797, 6.9903717, -23.0704117, 7.3563652, -29.8224449, 30.0607796
3: -19.9012680, 8.9243526, -20.5380898, 9.2507601, -29.1520195, 29.4624424
4: -20.2617760, 12.1930265, -20.7957764, 12.5690279, -32.8308029, 32.9888039
5: -14.9546680, 12.9803066, -15.4290638, 13.3535156, -28.3081818, 28.4093628
6: -16.0028496, 13.6762362, -16.4827061, 14.0766678, -30.0795135, 30.1589432
7: -18.0861378, 13.0791740, -18.6212692, 13.5060740, -31.5922012, 31.7004433
8: -20.8303032, 11.8382320, -21.4277573, 12.2034607, -33.0337639, 33.2659912
9: -14.4516153, 17.0915165, -14.8881588, 17.5661545, -32.0177689, 31.9796753

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1723257, upper bound: 27.1729935
time: 3.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1697128, upper bound: 27.1716974
time: 8.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -17.3379898, 13.1264954, -16.8317299, 12.7335157, -30.0714989, 29.9582253
1: -13.8477888, 11.6870747, -13.4129200, 11.3248215, -25.1726112, 25.0999908
2: -23.0851631, 7.3696795, -22.4660797, 6.9903717, -30.0755348, 29.8357582
3: -20.5558205, 9.2606688, -19.9012680, 8.9243526, -29.4801731, 29.1619301
4: -20.8129292, 12.5806046, -20.2617760, 12.1930265, -33.0059547, 32.8423805
5: -15.4483376, 13.3657217, -14.9546680, 12.9803066, -28.4286366, 28.3203869
6: -16.4954891, 14.0895319, -16.0028496, 13.6762362, -30.1717186, 30.0923767
7: -18.6347389, 13.5205002, -18.0861378, 13.0791740, -31.7139130, 31.6066360
8: -21.4465790, 12.2142963, -20.8303032, 11.8382320, -33.2848129, 33.0445976
9: -14.9008160, 17.5803413, -14.4516153, 17.0915165, -31.9923325, 32.0319557

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1729935, upper bound: 27.1742989
time: 7.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1716974, upper bound: 27.1727811
time: 7.25 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -17.3379898, 13.1264954, -17.3379898, 13.1264954, -30.4644852, 30.4644852
1: -13.8477888, 11.6870747, -13.8477888, 11.6870747, -25.5348625, 25.5348625
2: -23.0851631, 7.3696795, -23.0851631, 7.3696795, -30.4548397, 30.4548416
3: -20.5558205, 9.2606688, -20.5558205, 9.2606688, -29.8164902, 29.8164902
4: -20.8129292, 12.5806046, -20.8129292, 12.5806046, -33.3935318, 33.3935318
5: -15.4483376, 13.3657217, -15.4483376, 13.3657217, -28.8140526, 28.8140526
6: -16.4954891, 14.0895319, -16.4954891, 14.0895319, -30.5850182, 30.5850163
7: -18.6347389, 13.5205002, -18.6347389, 13.5205002, -32.1552391, 32.1552391
8: -21.4465790, 12.2142963, -21.4465790, 12.2142963, -33.6608734, 33.6608734
9: -14.9008160, 17.5803413, -14.9008160, 17.5803413, -32.4811516, 32.4811554

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752784, upper bound: 27.1878163
time: 9.15 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1716974, upper bound: 27.1866209
time: 13.11 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -16.8317299, 12.7335157, -17.8874741, 13.5268631, -30.3585873, 30.6209888
1: -13.4129200, 11.3248215, -14.2625885, 12.0275040, -25.4404182, 25.5874081
2: -22.4660797, 6.9903717, -23.8239956, 7.5214262, -29.9875050, 30.8143673
3: -19.9012680, 8.9243526, -21.1708260, 9.4986572, -29.3999214, 30.0951786
4: -20.2617760, 12.1930265, -21.4935303, 12.9550762, -33.2168465, 33.6865578
5: -14.9546680, 12.9803066, -15.9107056, 13.7824078, -28.7370758, 28.8910084
6: -16.0028496, 13.6762362, -17.0131073, 14.5267038, -30.5295448, 30.6893387
7: -18.0861378, 13.0791740, -19.2299156, 13.9158087, -32.0019417, 32.3090820
8: -20.8303032, 11.8382320, -22.1316624, 12.5717964, -33.4020996, 33.9698944
9: -14.4516153, 17.0915165, -15.3638668, 18.1342430, -32.5858574, 32.4553833

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1724571, upper bound: 27.1718093
time: 3.73 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1697928, upper bound: 27.1697680
time: 4.01 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -17.3238983, 13.1115246, -17.8874741, 13.5268631, -30.8507519, 30.9989986
1: -13.8358364, 11.6765242, -14.2625885, 12.0275040, -25.8633385, 25.9391041
2: -23.0704117, 7.3563652, -23.8239956, 7.5214262, -30.5918331, 31.1803608
3: -20.5380898, 9.2507601, -21.1708260, 9.4986572, -30.0367432, 30.4215832
4: -20.7957764, 12.5690279, -21.4935303, 12.9550762, -33.7508507, 34.0625572
5: -15.4290638, 13.3535156, -15.9107056, 13.7824078, -29.2114677, 29.2642155
6: -16.4827061, 14.0766678, -17.0131073, 14.5267038, -31.0094109, 31.0897675
7: -18.6212692, 13.5060740, -19.2299156, 13.9158087, -32.5370789, 32.7359848
8: -21.4277573, 12.2034607, -22.1316624, 12.5717964, -33.9995537, 34.3351212
9: -14.8881588, 17.5661545, -15.3638668, 18.1342430, -33.0223999, 32.9300232

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1718375, upper bound: 27.1743074
time: 6.85 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1697928, upper bound: 27.1727760
time: 4.10 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -16.8317299, 12.7335157, -18.7389374, 14.2121658, -31.0438957, 31.4724541
1: -13.4129200, 11.3248215, -14.9956312, 12.6541805, -26.0671005, 26.3204536
2: -22.4660797, 6.9903717, -24.8272076, 8.2060719, -30.6721516, 31.8175793
3: -19.9012680, 8.9243526, -22.2329102, 10.0517826, -29.9530487, 31.1572609
4: -20.2617760, 12.1930265, -22.4428482, 13.6119614, -33.8737373, 34.6358757
5: -14.9546680, 12.9803066, -16.7553368, 14.4334555, -29.3881226, 29.7356434
6: -16.0028496, 13.6762362, -17.8754063, 15.2532148, -31.2560616, 31.5516434
7: -18.0861378, 13.0791740, -20.1374226, 14.6879034, -32.7740364, 33.2165985
8: -20.8303032, 11.8382320, -23.2031250, 13.2296448, -34.0599480, 35.0413589
9: -14.4516153, 17.0915165, -16.1392937, 18.9533577, -33.4049721, 33.2308121

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1724571, upper bound: 27.1732254
time: 6.84 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1697928, upper bound: 27.1719930
time: 4.22 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -17.3379898, 13.1264954, -18.7389374, 14.2121658, -31.5501556, 31.8654327
1: -13.8477888, 11.6870747, -14.9956312, 12.6541805, -26.5019684, 26.6827049
2: -23.0851631, 7.3696795, -24.8272076, 8.2060719, -31.2912350, 32.1968842
3: -20.5558205, 9.2606688, -22.2329102, 10.0517826, -30.6076031, 31.4935760
4: -20.8129292, 12.5806046, -22.4428482, 13.6119614, -34.4248886, 35.0234528
5: -15.4483376, 13.3657217, -16.7553368, 14.4334555, -29.8817902, 30.1210594
6: -16.4954891, 14.0895319, -17.8754063, 15.2532148, -31.7486992, 31.9649353
7: -18.6347389, 13.5205002, -20.1374226, 14.6879034, -33.3226433, 33.6579208
8: -21.4465790, 12.2142963, -23.2031250, 13.2296448, -34.6762238, 35.4174156
9: -14.9008160, 17.5803413, -16.1392937, 18.9533577, -33.8541718, 33.7196312

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1724571, upper bound: 27.1880622
time: 6.91 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1697928, upper bound: 27.1870516
time: 6.10 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -17.8874741, 13.5268631, -16.8317299, 12.7335157, -30.6209908, 30.3585892
1: -14.2625885, 12.0275040, -13.4129200, 11.3248215, -25.5874062, 25.4404202
2: -23.8239956, 7.5214262, -22.4660797, 6.9903717, -30.8143673, 29.9875031
3: -21.1708260, 9.4986572, -19.9012680, 8.9243526, -30.0951786, 29.3999176
4: -21.4935303, 12.9550762, -20.2617760, 12.1930265, -33.6865578, 33.2168503
5: -15.9107056, 13.7824078, -14.9546680, 12.9803066, -28.8910046, 28.7370758
6: -17.0131073, 14.5267038, -16.0028496, 13.6762362, -30.6893425, 30.5295486
7: -19.2299156, 13.9158087, -18.0861378, 13.0791740, -32.3090820, 32.0019455
8: -22.1316624, 12.5717964, -20.8303032, 11.8382320, -33.9698944, 33.4020996
9: -15.3638668, 18.1342430, -14.4516153, 17.0915165, -32.4553833, 32.5858574

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1718093, upper bound: 27.1724571
time: 4.47 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1697680, upper bound: 27.1697928
time: 3.47 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -17.8874741, 13.5268631, -17.3238983, 13.1115246, -30.9989986, 30.8507538
1: -14.2625885, 12.0275040, -13.8358364, 11.6765242, -25.9391060, 25.8633404
2: -23.8239956, 7.5214262, -23.0704117, 7.3563652, -31.1803608, 30.5918350
3: -21.1708260, 9.4986572, -20.5380898, 9.2507601, -30.4215813, 30.0367470
4: -21.4935303, 12.9550762, -20.7957764, 12.5690279, -34.0625572, 33.7508545
5: -15.9107056, 13.7824078, -15.4290638, 13.3535156, -29.2642174, 29.2114716
6: -17.0131073, 14.5267038, -16.4827061, 14.0766678, -31.0897675, 31.0094109
7: -19.2299156, 13.9158087, -18.6212692, 13.5060740, -32.7359848, 32.5370789
8: -22.1316624, 12.5717964, -21.4277573, 12.2034607, -34.3351212, 33.9995537
9: -15.3638668, 18.1342430, -14.8881588, 17.5661545, -32.9300232, 33.0223999

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1723577, upper bound: 27.1718374
time: 14.29 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1697680, upper bound: 27.1717263
time: 6.18 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -18.7389374, 14.2121658, -16.8317299, 12.7335157, -31.4724541, 31.0438957
1: -14.9956312, 12.6541805, -13.4129200, 11.3248215, -26.3204536, 26.0671005
2: -24.8272076, 8.2060719, -22.4660797, 6.9903717, -31.8175774, 30.6721516
3: -22.2329102, 10.0517826, -19.9012680, 8.9243526, -31.1572609, 29.9530487
4: -22.4428482, 13.6119614, -20.2617760, 12.1930265, -34.6358757, 33.8737373
5: -16.7553368, 14.4334555, -14.9546680, 12.9803066, -29.7356396, 29.3881226
6: -17.8754063, 15.2532148, -16.0028496, 13.6762362, -31.5516434, 31.2560654
7: -20.1374226, 14.6879034, -18.0861378, 13.0791740, -33.2165985, 32.7740402
8: -23.2031250, 13.2296448, -20.8303032, 11.8382320, -35.0413589, 34.0599480
9: -16.1392937, 18.9533577, -14.4516153, 17.0915165, -33.2308121, 33.4049721

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1732254, upper bound: 27.1723257
time: 15.49 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1719930, upper bound: 27.1732357
time: 11.27 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -18.7389374, 14.2121658, -17.3379898, 13.1264954, -31.8654327, 31.5501556
1: -14.9956312, 12.6541805, -13.8477888, 11.6870747, -26.6827011, 26.5019684
2: -24.8272076, 8.2060719, -23.0851631, 7.3696795, -32.1968842, 31.2912350
3: -22.2329102, 10.0517826, -20.5558205, 9.2606688, -31.4935780, 30.6076031
4: -22.4428482, 13.6119614, -20.8129292, 12.5806046, -35.0234528, 34.4248886
5: -16.7553368, 14.4334555, -15.4483376, 13.3657217, -30.1210556, 29.8817902
6: -17.8754063, 15.2532148, -16.4954891, 14.0895319, -31.9649353, 31.7487011
7: -20.1374226, 14.6879034, -18.6347389, 13.5205002, -33.6579208, 33.3226433
8: -23.2031250, 13.2296448, -21.4465790, 12.2142963, -35.4174156, 34.6762238
9: -16.1392937, 18.9533577, -14.9008160, 17.5803413, -33.7196312, 33.8541679

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1732254, upper bound: 27.1887686
time: 8.49 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1719930, upper bound: 27.1873746
time: 33.25 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -17.8874741, 13.5268631, -17.8874741, 13.5268631, -31.4143353, 31.4143372
1: -14.2625885, 12.0275040, -14.2625885, 12.0275040, -26.2900925, 26.2900867
2: -23.8239956, 7.5214262, -23.8239956, 7.5214262, -31.3454208, 31.3454208
3: -21.1708260, 9.4986572, -21.1708260, 9.4986572, -30.6694832, 30.6694813
4: -21.4935303, 12.9550762, -21.4935303, 12.9550762, -34.4486046, 34.4486084
5: -15.9107056, 13.7824078, -15.9107056, 13.7824078, -29.6931114, 29.6931133
6: -17.0131073, 14.5267038, -17.0131073, 14.5267038, -31.5398064, 31.5398026
7: -19.2299156, 13.9158087, -19.2299156, 13.9158087, -33.1457253, 33.1457214
8: -22.1316624, 12.5717964, -22.1316624, 12.5717964, -34.7034607, 34.7034607
9: -15.3638668, 18.1342430, -15.3638668, 18.1342430, -33.4981079, 33.4981079

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1612729, upper bound: 27.1586973
time: 6.43 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1555969, upper bound: 27.1555969
time: 2.34 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -18.7153397, 14.1912203, -17.8874741, 13.5268631, -32.2422028, 32.0786934
1: -14.9776440, 12.6317577, -14.2625885, 12.0275040, -27.0051479, 26.8943443
2: -24.8011169, 8.1786776, -23.8239956, 7.5214262, -32.3225403, 32.0026703
3: -22.1939545, 10.0299835, -21.1708260, 9.4986572, -31.6926003, 31.2008076
4: -22.4154968, 13.5932827, -21.4935303, 12.9550762, -35.3705750, 35.0868111
5: -16.7348785, 14.4133339, -15.9107056, 13.7824078, -30.5172863, 30.3240299
6: -17.8531113, 15.2300110, -17.0131073, 14.5267038, -32.3798141, 32.2431183
7: -20.1147156, 14.6646862, -19.2299156, 13.9158087, -34.0305252, 33.8945999
8: -23.1734371, 13.2032824, -22.1316624, 12.5717964, -35.7452316, 35.3349457
9: -16.1142273, 18.9304619, -15.3638668, 18.1342430, -34.2484703, 34.2943268

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 58

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1788006, upper bound: 27.1812706
time: 4.30 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1771024, upper bound: 27.1799068
time: 14.77 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -17.8874741, 13.5268631, -18.7389374, 14.2121658, -32.0996399, 32.2658005
1: -14.2625885, 12.0275040, -14.9956312, 12.6541805, -26.9167671, 27.0231323
2: -23.8239956, 7.5214262, -24.8272076, 8.2060719, -32.0300674, 32.3486328
3: -21.1708260, 9.4986572, -22.2329102, 10.0517826, -31.2226086, 31.7315636
4: -21.4935303, 12.9550762, -22.4428482, 13.6119614, -35.1054916, 35.3979263
5: -15.9107056, 13.7824078, -16.7553368, 14.4334555, -30.3441582, 30.5377445
6: -17.0131073, 14.5267038, -17.8754063, 15.2532148, -32.2663193, 32.4021072
7: -19.2299156, 13.9158087, -20.1374226, 14.6879034, -33.9178123, 34.0532303
8: -22.1316624, 12.5717964, -23.2031250, 13.2296448, -35.3613052, 35.7749214
9: -15.3638668, 18.1342430, -16.1392937, 18.9533577, -34.3172226, 34.2735291

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 58

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1784344, upper bound: 27.1806819
time: 9.60 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1771024, upper bound: 27.1787451
time: 6.29 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -18.7389374, 14.2121658, -18.7389374, 14.2121658, -32.9511032, 32.9511032
1: -14.9956312, 12.6541805, -14.9956312, 12.6541805, -27.6498108, 27.6498108
2: -24.8272076, 8.2060719, -24.8272076, 8.2060719, -33.0332794, 33.0332794
3: -22.2329102, 10.0517826, -22.2329102, 10.0517826, -32.2846909, 32.2846909
4: -22.4428482, 13.6119614, -22.4428482, 13.6119614, -36.0548096, 36.0548096
5: -16.7553368, 14.4334555, -16.7553368, 14.4334555, -31.1887932, 31.1887932
6: -17.8754063, 15.2532148, -17.8754063, 15.2532148, -33.1286201, 33.1286201
7: -20.1374226, 14.6879034, -20.1374226, 14.6879034, -34.8253250, 34.8253250
8: -23.2031250, 13.2296448, -23.2031250, 13.2296448, -36.4327660, 36.4327698
9: -16.1392937, 18.9533577, -16.1392937, 18.9533577, -35.0926437, 35.0926437

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1587670, upper bound: 27.1843221
time: 8.44 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1555969, upper bound: 27.1555969
time: 25.99 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 44.27 seconds
IS_A1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1723257, upper bound: 27.1717367
IS_A1_B1_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1697128, upper bound: 27.1697128
IS_A1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1723257, upper bound: 27.1729935
IS_A1_B1_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1697128, upper bound: 27.1716974
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1729935, upper bound: 27.1742989
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1716974, upper bound: 27.1727811
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1752784, upper bound: 27.1878163
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1716974, upper bound: 27.1866209
IS_A1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1724571, upper bound: 27.1718093
IS_A1_B2_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1697928, upper bound: 27.1697680
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1718375, upper bound: 27.1743074
IS_A1_B2_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1697928, upper bound: 27.1727760
IS_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1724571, upper bound: 27.1732254
IS_A1_B2_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1697928, upper bound: 27.1719930
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1724571, upper bound: 27.1880622
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1697928, upper bound: 27.1870516
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1718093, upper bound: 27.1724571
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1697680, upper bound: 27.1697928
IS_A2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1723577, upper bound: 27.1718374
IS_A2_B1_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1697680, upper bound: 27.1717263
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1732254, upper bound: 27.1723257
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1719930, upper bound: 27.1732357
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1732254, upper bound: 27.1887686
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1719930, upper bound: 27.1873746
IS_A2_B2_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1612729, upper bound: 27.1586973
IS_A2_B2_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1555969, upper bound: 27.1555969
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1788006, upper bound: 27.1812706
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1771024, upper bound: 27.1799068
IS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1784344, upper bound: 27.1806819
IS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1771024, upper bound: 27.1787451
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1587670, upper bound: 27.1843221
IS_A2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 44.27
Output dim: 2, lower bound: -27.1555969, upper bound: 27.1555969

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -16.8125973, 12.7328911, -16.8016129, 12.7111797, -29.5237770, 29.5345001
1: -13.4218760, 11.3385963, -13.3884916, 11.3049088, -24.7267818, 24.7270889
2: -22.3968735, 7.1381764, -22.4261818, 6.9775701, -29.3744392, 29.5643578
3: -19.9246082, 8.9845314, -19.8650570, 8.9087105, -28.8333187, 28.8495884
4: -20.1898422, 12.2064762, -20.2259445, 12.1716232, -32.3614655, 32.4324188
5: -14.9760876, 12.9663172, -14.9279737, 12.9573383, -27.9334221, 27.8942909
6: -15.9941921, 13.6665220, -15.9741259, 13.6519890, -29.6461811, 29.6406479
7: -18.0708981, 13.1129217, -18.0536880, 13.0559940, -31.1268921, 31.1666069
8: -20.7974510, 11.8515377, -20.7930851, 11.8175049, -32.6149559, 32.6446228
9: -14.4500961, 17.0535355, -14.4258394, 17.0612526, -31.5113487, 31.4793739

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1716974, upper bound: 27.1727811
time: 5.38 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1716974, upper bound: 27.1727811
time: 4.17 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -17.3072186, 13.1034393, -16.8125973, 12.7328911, -30.0401077, 29.9160366
1: -13.8228397, 11.6666660, -13.4218760, 11.3385963, -25.1614323, 25.0885353
2: -23.0448456, 7.3561068, -22.3968735, 7.1381764, -30.1830215, 29.7529736
3: -20.5188541, 9.2444859, -19.9246082, 8.9845314, -29.5033798, 29.1690941
4: -20.7764225, 12.5586872, -20.1898422, 12.2064762, -32.9828987, 32.7485275
5: -15.4206753, 13.3423338, -14.9760876, 12.9663172, -28.3869934, 28.3184204
6: -16.4661255, 14.0647621, -15.9941921, 13.6665220, -30.1326485, 30.0589542
7: -18.6017113, 13.4966183, -18.0708981, 13.1129217, -31.7146339, 31.5675144
8: -21.4085598, 12.1930447, -20.7974510, 11.8515377, -33.2600975, 32.9904938
9: -14.8744125, 17.5494862, -14.4500961, 17.0535355, -31.9279480, 31.9995823

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1866107, upper bound: 27.1866209
time: 8.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1866107, upper bound: 27.1866209
time: 5.05 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 15.26 seconds
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 15.26
Output dim: 2, lower bound: -27.1716974, upper bound: 27.1727811
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 15.26
Output dim: 2, lower bound: -27.1716974, upper bound: 27.1727811
IS_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 15.26
Output dim: 2, lower bound: -27.1866107, upper bound: 27.1866209
IS_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 15.26
Output dim: 2, lower bound: -27.1866107, upper bound: 27.1866209
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 15.26
Output dim: 2, lower bound: -27.1716974, upper bound: 27.1866209
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.26
Output dim: 2, lower bound: -27.1718375, upper bound: 27.1743074
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 15.26
Output dim: 2, lower bound: -27.1724571, upper bound: 27.1880622
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 15.26
Output dim: 2, lower bound: -27.1697928, upper bound: 27.1870516
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.26
Output dim: 2, lower bound: -27.1732254, upper bound: 27.1887686
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.26
Output dim: 2, lower bound: -27.1719930, upper bound: 27.1873746
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 15.26
Output dim: 2, lower bound: -27.1788006, upper bound: 27.1812706
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 15.26
Output dim: 2, lower bound: -27.1771024, upper bound: 27.1799068
IS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 15.26
Output dim: 2, lower bound: -27.1784344, upper bound: 27.1806819
IS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 15.26
Output dim: 2, lower bound: -27.1771024, upper bound: 27.1787451
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.26
Output dim: 2, lower bound: -27.1587670, upper bound: 27.1843221
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=36.862388610839844
rel_dist={2: [-27.20050538908776, 27.200505386462353]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1945032, upper bound: 27.1945638
time: 6.26 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1942730, upper bound: 27.1942730
time: 9.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.13 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.13
Output dim: 2, lower bound: -27.1945032, upper bound: 27.1945638
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.13
Output dim: 2, lower bound: -27.1942730, upper bound: 27.1942730

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -18.7070465, 14.2073870, -19.3088303, 14.6746416, -33.3816872, 33.5162163
1: -14.9808378, 12.6602945, -15.4782963, 13.0825186, -28.0633564, 28.1385918
2: -24.7259140, 8.3407669, -25.4674625, 8.7117224, -33.4376335, 33.8082275
3: -22.1942329, 10.0750408, -22.9273834, 10.4317923, -32.6260262, 33.0024261
4: -22.3834515, 13.6224461, -23.0868225, 14.0752277, -36.4586792, 36.7092628
5: -16.7453861, 14.4127302, -17.3073273, 14.8761616, -31.6215477, 31.7200584
6: -17.8827705, 15.2430191, -18.4703751, 15.7486954, -33.6314659, 33.7133942
7: -20.0948524, 14.7141514, -20.7334595, 15.2172575, -35.3121033, 35.4476089
8: -23.1707306, 13.2475548, -23.9217758, 13.6934757, -36.8642044, 37.1693306
9: -16.1366158, 18.8929558, -16.6736889, 19.4781876, -35.6148033, 35.5666389

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1878221, upper bound: 27.1879828
time: 5.98 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1875506, upper bound: 27.1875729
time: 9.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -19.2869339, 14.6154661, -17.8986320, 13.5895996, -32.8765297, 32.5140991
1: -15.4296846, 12.9933205, -14.3172808, 12.1016006, -27.5312805, 27.3106003
2: -25.5673485, 8.3691521, -23.7062817, 7.8937759, -33.4611206, 32.0754242
3: -22.8849373, 10.3076086, -21.2215061, 9.6232519, -32.5081902, 31.5291138
4: -23.0926991, 13.9870644, -21.4299259, 13.0251055, -36.1178055, 35.4169922
5: -17.2289028, 14.8471441, -16.0056839, 13.7950230, -31.0239258, 30.8528214
6: -18.3952656, 15.6864824, -17.0854263, 14.5785732, -32.9738350, 32.7719078
7: -20.7276421, 15.0893497, -19.2365494, 14.0593576, -34.7869949, 34.3258972
8: -23.8640194, 13.5947933, -22.1557903, 12.6623878, -36.5264053, 35.7505798
9: -16.5932846, 19.5126495, -15.4240980, 18.0965862, -34.6898727, 34.9367485

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1875950, upper bound: 27.1877668
time: 20.87 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1872318, upper bound: 27.1872318
time: 4.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.61 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 26.61
Output dim: 2, lower bound: -27.1878221, upper bound: 27.1879828
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 26.61
Output dim: 2, lower bound: -27.1875506, upper bound: 27.1875729
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 26.61
Output dim: 2, lower bound: -27.1875950, upper bound: 27.1877668
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 26.61
Output dim: 2, lower bound: -27.1872318, upper bound: 27.1872318

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -18.0496902, 13.7092524, -18.8375130, 14.3176918, -32.3673820, 32.5467644
1: -14.4434052, 12.2181559, -15.0935631, 12.7659950, -27.2093964, 27.3117180
2: -23.8703899, 8.0296154, -24.8561592, 8.4870710, -32.3574600, 32.8857727
3: -21.3990936, 9.7237320, -22.3584518, 10.1788158, -31.5779095, 32.0821762
4: -21.6026363, 13.1484632, -22.5281639, 13.7352028, -35.3378296, 35.6766281
5: -16.1501637, 13.9117155, -16.8803158, 14.5176954, -30.6678581, 30.7920303
6: -17.2502041, 14.7074261, -18.0177155, 15.3641739, -32.6143799, 32.7251358
7: -19.3926086, 14.1984921, -20.2307663, 14.8478889, -34.2404976, 34.4292603
8: -22.3560543, 12.7834110, -23.3396225, 13.3600349, -35.7160873, 36.1230240
9: -15.5682907, 18.2338905, -16.2667923, 19.0070953, -34.5753860, 34.5006828

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1865498, upper bound: 27.1867072
time: 8.02 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1864217, upper bound: 27.1865867
time: 5.91 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -21.8739471, 16.6404076, -18.4833889, 14.0487461, -35.9226913, 35.1237946
1: -17.6003647, 14.8603287, -14.8033781, 12.5277739, -30.1281357, 29.6637058
2: -28.6870041, 10.1499567, -24.3936081, 8.3208323, -37.0078354, 34.5435638
3: -26.0405083, 11.9036217, -21.9313335, 9.9893713, -36.0298767, 33.8349533
4: -26.1223450, 15.9891672, -22.1066475, 13.4789877, -39.6013336, 38.0958099
5: -19.6810131, 16.8512077, -16.5593948, 14.2468624, -33.9278755, 33.4106026
6: -20.9585266, 17.8922787, -17.6776619, 15.0742722, -36.0327988, 35.5699387
7: -23.4616947, 17.3213291, -19.8518181, 14.5701828, -38.0318756, 37.1731491
8: -27.1139717, 15.5623856, -22.9005775, 13.1092472, -40.2232208, 38.4629593
9: -18.9394379, 22.0048752, -15.9606514, 18.6513977, -37.5908279, 37.9655266

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 160

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1826633, upper bound: 27.1825056
time: 7.65 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1857578, upper bound: 27.1857704
time: 5.68 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -18.6701317, 14.1499434, -17.4451675, 13.2487478, -31.9188766, 31.5951099
1: -14.9267998, 12.5791245, -13.9484596, 11.7979202, -26.7247200, 26.5275841
2: -24.7600594, 8.0843678, -23.1157837, 7.6831608, -32.4432220, 31.2001495
3: -22.1416359, 9.9828987, -20.6759758, 9.3834467, -31.5250816, 30.6588745
4: -22.3601608, 13.5443487, -20.8935356, 12.6995449, -35.0597076, 34.4378815
5: -16.6714859, 14.3785963, -15.5970354, 13.4514503, -30.1229362, 29.9756317
6: -17.8028870, 15.1854296, -16.6496315, 14.2120705, -32.0149574, 31.8350601
7: -20.0662460, 14.6073408, -18.7519836, 13.7053680, -33.7716141, 33.3593216
8: -23.0988045, 13.1624870, -21.5944748, 12.3454084, -35.4442101, 34.7569618
9: -16.0616951, 18.8933983, -15.0341082, 17.6433067, -33.7050018, 33.9275017

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1862853, upper bound: 27.1864584
time: 11.89 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1862288, upper bound: 27.1863957
time: 10.78 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -22.1367626, 16.7690525, -17.1154099, 13.0001431, -35.1369019, 33.8844604
1: -17.7810516, 14.9602280, -13.6796942, 11.5764914, -29.3575401, 28.6399193
2: -29.1511688, 9.9323092, -22.6836243, 7.5325146, -36.6836777, 32.6159325
3: -26.3446465, 11.8998976, -20.2796364, 9.2108240, -35.5554695, 32.1795311
4: -26.4421215, 16.0792198, -20.5015640, 12.4623137, -38.9044304, 36.5807838
5: -19.8488731, 17.0221119, -15.2992477, 13.2006178, -33.0494843, 32.3213577
6: -21.1606941, 18.0247803, -16.3329220, 13.9443207, -35.1050148, 34.3577042
7: -23.7609501, 17.4115601, -18.3982544, 13.4479446, -37.2088928, 35.8098145
8: -27.4112301, 15.6264801, -21.1849098, 12.1147099, -39.5259361, 36.8113899
9: -19.0900517, 22.3106461, -14.7504168, 17.3117428, -36.4017944, 37.0610619

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1859583, upper bound: 27.1859462
time: 6.51 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1858554, upper bound: 27.1858554
time: 4.87 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 12.67 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 12.67
Output dim: 2, lower bound: -27.1865498, upper bound: 27.1867072
IS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 12.67
Output dim: 2, lower bound: -27.1864217, upper bound: 27.1865867
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 12.67
Output dim: 2, lower bound: -27.1826633, upper bound: 27.1825056
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 12.67
Output dim: 2, lower bound: -27.1857578, upper bound: 27.1857704
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 12.67
Output dim: 2, lower bound: -27.1862853, upper bound: 27.1864584
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 12.67
Output dim: 2, lower bound: -27.1862288, upper bound: 27.1863957
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 12.67
Output dim: 2, lower bound: -27.1859583, upper bound: 27.1859462
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 12.67
Output dim: 2, lower bound: -27.1858554, upper bound: 27.1858554

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: -17.4389954, 13.2485342, -18.3568802, 13.9532681, -31.3922520, 31.6054153
1: -13.9453535, 11.8077698, -14.7009277, 12.4421768, -26.3875294, 26.5086975
2: -23.0766277, 7.7422113, -24.2332478, 8.2579327, -31.3345604, 31.9754486
3: -20.6632423, 9.3974886, -21.7781982, 9.9199486, -30.5831871, 31.1756859
4: -20.8789978, 12.7080936, -21.9579010, 13.3884754, -34.2674713, 34.6659927
5: -15.5988035, 13.4459724, -16.4450855, 14.1508808, -29.7496834, 29.8910522
6: -16.6626015, 14.2114401, -17.5554523, 14.9715328, -31.6341267, 31.7668915
7: -18.7401390, 13.7192526, -19.7176342, 14.4700317, -33.2101707, 33.4368858
8: -21.5968056, 12.3544559, -22.7439003, 13.0202465, -34.6170502, 35.0983505
9: -15.0410137, 17.6224842, -15.8505058, 18.5264664, -33.5674820, 33.4729881

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1808649, upper bound: 27.1811820
time: 7.66 seconds

## Relational analysis of IS_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1847540, upper bound: 27.1849212
time: 6.75 seconds

## BFS IS instance: IS_A1_A1_A2

### Backsubstitution after applying IS history:
0: -19.0009060, 14.4342642, -18.3850937, 13.9741659, -32.9750710, 32.8193550
1: -15.2446346, 12.8790569, -14.7242165, 12.4609966, -27.7056313, 27.6032734
2: -25.0779572, 8.5531616, -24.2708588, 8.2703972, -33.3483505, 32.8240204
3: -22.5419617, 10.2437992, -21.8121185, 9.9348373, -32.4767990, 32.0559158
4: -22.7319412, 13.8577251, -21.9913292, 13.4082527, -36.1401863, 35.8490524
5: -17.0252495, 14.6416817, -16.4700394, 14.1722202, -31.1974640, 31.1117172
6: -18.1843605, 15.5032005, -17.5821285, 14.9945202, -33.1788788, 33.0853271
7: -20.4171467, 14.9719915, -19.7479916, 14.4917450, -34.9088898, 34.7199821
8: -23.5458031, 13.4744453, -22.7787437, 13.0396843, -36.5854874, 36.2531891
9: -16.4069271, 19.1807270, -15.8747225, 18.5547142, -34.9616394, 35.0554428

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_A2_A1

### Relational analysis result of IS_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1812029, upper bound: 27.1811914
time: 5.71 seconds

## Relational analysis of IS_A1_A1_A2_A2

### Relational analysis result of IS_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1846268, upper bound: 27.1848081
time: 4.25 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -18.8637123, 14.2866011, -17.0295906, 12.8999367, -31.7636395, 31.3161926
1: -15.1029530, 12.7276201, -13.5989208, 11.4832850, -26.5862389, 26.3265419
2: -25.0138378, 8.1777706, -22.6573658, 7.2614274, -32.2752647, 30.8351364
3: -22.4000797, 10.0954227, -20.1793022, 9.1088333, -31.5089130, 30.2747231
4: -22.6186943, 13.7019386, -20.4331894, 12.3625965, -34.9812927, 34.1351280
5: -16.8572044, 14.5314045, -15.1752987, 13.1316776, -29.9888763, 29.7067032
6: -17.9702358, 15.3452091, -16.2098274, 13.8408308, -31.8110657, 31.5550365
7: -20.2715836, 14.7468452, -18.3005180, 13.2930508, -33.5646362, 33.0473633
8: -23.3543205, 13.2880306, -21.0618572, 12.0057840, -35.3601036, 34.3498840
9: -16.2309456, 19.0928326, -14.6380053, 17.2628174, -33.4937592, 33.7308350

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1764457, upper bound: 27.1762146
time: 5.52 seconds

## Relational analysis of IS_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1762355, upper bound: 27.1760618
time: 6.74 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -20.5492687, 15.5847559, -17.5715141, 13.3333549, -33.8826218, 33.1562691
1: -16.4946156, 13.9012356, -14.0489149, 11.8710451, -28.3656616, 27.9501495
2: -27.1012230, 9.2004938, -23.2975731, 7.6750941, -34.7763176, 32.4980621
3: -24.4268398, 11.0733433, -20.8260002, 9.4349670, -33.8617973, 31.8993435
4: -24.5807343, 14.9554234, -21.0522728, 12.7762804, -37.3570137, 36.0076904
5: -18.4190598, 15.8217611, -15.7001410, 13.5458241, -31.9648819, 31.5219002
6: -19.6294575, 16.7577286, -16.7611961, 14.3056993, -33.9351578, 33.5189171
7: -22.0608292, 16.1571274, -18.8856583, 13.7773209, -35.8381500, 35.0427856
8: -25.4551849, 14.5331306, -21.7507954, 12.4158096, -37.8709946, 36.2839241
9: -17.7272854, 20.7359028, -15.1328440, 17.7801666, -35.5074501, 35.8687477

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1822118, upper bound: 27.1823657
time: 9.62 seconds

## Relational analysis of IS_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1822118, upper bound: 27.1823657
time: 4.97 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: -18.0846367, 13.7076359, -16.9919128, 12.9071960, -30.9918327, 30.6995468
1: -14.4500580, 12.1860752, -13.5797625, 11.4930067, -25.9430656, 25.7658348
2: -23.9937172, 7.8153257, -22.5245018, 7.4725494, -31.4662666, 30.3398228
3: -21.4369202, 9.6748848, -20.1300964, 9.1435757, -30.5804958, 29.8049755
4: -21.6642704, 13.1243896, -20.3562527, 12.3737717, -34.0380402, 33.4806328
5: -16.1416283, 13.9335699, -15.1879826, 13.1068878, -29.2485123, 29.1215515
6: -17.2390594, 14.7095242, -16.2136154, 13.8440523, -31.0831108, 30.9231396
7: -19.4378281, 14.1497993, -18.2668953, 13.3506336, -32.7884598, 32.4166946
8: -22.3695908, 12.7532883, -21.0308456, 12.0276852, -34.3972778, 33.7841339
9: -15.5566082, 18.3049221, -14.6432743, 17.1885128, -32.7451172, 32.9481964

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1806656, upper bound: 27.1810379
time: 6.99 seconds

## Relational analysis of IS_A2_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1845058, upper bound: 27.1846959
time: 11.93 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -19.4598961, 14.7442169, -17.0295944, 12.9355373, -32.3954315, 31.7738094
1: -15.5804138, 13.1199989, -13.6108036, 11.5183659, -27.0987778, 26.7308025
2: -25.7717819, 8.4924726, -22.5746574, 7.4893923, -33.2611732, 31.0671310
3: -23.0847168, 10.3997860, -20.1756706, 9.1634502, -32.2481689, 30.5754528
4: -23.2974815, 14.1194706, -20.4011078, 12.4005566, -35.6980362, 34.5205765
5: -17.3897858, 14.9743509, -15.2217827, 13.1356544, -30.5254402, 30.1961327
6: -18.5772610, 15.8360701, -16.2496719, 13.8747873, -32.4520493, 32.0857391
7: -20.9164619, 15.2364922, -18.3074532, 13.3799791, -34.2964401, 33.5439453
8: -24.0833015, 13.7217646, -21.0777874, 12.0538807, -36.1371803, 34.7995529
9: -16.7505608, 19.6791077, -14.6757154, 17.2265320, -33.9770927, 34.3548241

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1806073, upper bound: 27.1809888
time: 7.73 seconds

## Relational analysis of IS_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1844524, upper bound: 27.1846309
time: 7.92 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -21.6856537, 16.4292736, -16.5214653, 12.5517998, -34.2374535, 32.9507370
1: -17.4128304, 14.6568508, -13.1962776, 11.1763058, -28.5891361, 27.8531265
2: -28.5634117, 9.7223310, -21.9079590, 7.2568073, -35.8202133, 31.6302872
3: -25.8006477, 11.6597204, -19.5638580, 8.8962078, -34.6968536, 31.2235794
4: -25.9074078, 15.7547684, -19.7964325, 12.0356102, -37.9430161, 35.5511971
5: -19.4418831, 16.6786556, -14.7627544, 12.7487440, -32.1906204, 31.4414082
6: -20.7269325, 17.6582813, -15.7614975, 13.4612236, -34.1881485, 33.4197769
7: -23.2784061, 17.0577812, -17.7620316, 12.9829597, -36.2613678, 34.8198128
8: -26.8509083, 15.3096848, -20.4457207, 11.6979370, -38.5488396, 35.7554016
9: -18.7007504, 21.8583603, -14.2375259, 16.7153893, -35.4161377, 36.0958786

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1805547, upper bound: 27.1803313
time: 8.29 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1841891, upper bound: 27.1841779
time: 6.46 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -21.6990299, 16.4393158, -17.9285736, 13.6090384, -35.3080673, 34.3678894
1: -17.4239197, 14.6659060, -14.3553772, 12.1347466, -29.5586662, 29.0212822
2: -28.5813484, 9.7283764, -23.7183533, 7.9588652, -36.5402145, 33.4467278
3: -25.8169250, 11.6669750, -21.2519302, 9.6420584, -35.4589806, 32.9189072
4: -25.9231491, 15.7641068, -21.4620171, 13.0575151, -38.9806633, 37.2261238
5: -19.4537659, 16.6889229, -16.0383263, 13.8129511, -33.2667122, 32.7272491
6: -20.7396412, 17.6691818, -17.1306858, 14.6135721, -35.3532143, 34.7998657
7: -23.2925892, 17.0682564, -19.2710457, 14.0979853, -37.3905716, 36.3393021
8: -26.8675385, 15.3189411, -22.1980152, 12.6918297, -39.5593643, 37.5169525
9: -18.7122631, 21.8718185, -15.4595413, 18.1175976, -36.8298607, 37.3313522

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1804662, upper bound: 27.1802297
time: 5.70 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1840856, upper bound: 27.1840856
time: 6.16 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 13.14 seconds
IS_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.14
Output dim: 2, lower bound: -27.1808649, upper bound: 27.1811820
IS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.14
Output dim: 2, lower bound: -27.1847540, upper bound: 27.1849212
IS_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 13.14
Output dim: 2, lower bound: -27.1812029, upper bound: 27.1811914
IS_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 13.14
Output dim: 2, lower bound: -27.1846268, upper bound: 27.1848081
IS_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.14
Output dim: 2, lower bound: -27.1764457, upper bound: 27.1762146
IS_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.14
Output dim: 2, lower bound: -27.1762355, upper bound: 27.1760618
IS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.14
Output dim: 2, lower bound: -27.1822118, upper bound: 27.1823657
IS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.14
Output dim: 2, lower bound: -27.1822118, upper bound: 27.1823657
IS_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.14
Output dim: 2, lower bound: -27.1806656, upper bound: 27.1810379
IS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.14
Output dim: 2, lower bound: -27.1845058, upper bound: 27.1846959
IS_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.14
Output dim: 2, lower bound: -27.1806073, upper bound: 27.1809888
IS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.14
Output dim: 2, lower bound: -27.1844524, upper bound: 27.1846309
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 13.14
Output dim: 2, lower bound: -27.1805547, upper bound: 27.1803313
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 13.14
Output dim: 2, lower bound: -27.1841891, upper bound: 27.1841779
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 13.14
Output dim: 2, lower bound: -27.1804662, upper bound: 27.1802297
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 13.14
Output dim: 2, lower bound: -27.1840856, upper bound: 27.1840856

## BFS IS instance: IS_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -16.1827698, 12.2596521, -15.9500370, 12.0859213, -28.2686882, 28.2096844
1: -12.9108791, 10.9160032, -12.7246008, 10.7644367, -23.6753159, 23.6406021
2: -21.5635204, 6.8478303, -21.2724228, 6.7345991, -28.2981186, 28.1202526
3: -19.1565819, 8.6530619, -18.8845596, 8.5294695, -27.6860504, 27.5376205
4: -19.4332428, 11.7521505, -19.1651230, 11.5869751, -31.0202179, 30.9172707
5: -14.4059429, 12.4874907, -14.1994801, 12.3126621, -26.7186050, 26.6869698
6: -15.4001598, 13.1554546, -15.1744661, 12.9703503, -28.3705101, 28.3299141
7: -17.3893795, 12.6234970, -17.1411190, 12.4389362, -29.8283157, 29.7646160
8: -20.0123291, 11.4165421, -19.7302132, 11.2552385, -31.2675667, 31.1467552
9: -13.9069843, 16.4191551, -13.7081242, 16.1917610, -30.0987453, 30.1272793

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754565, upper bound: 27.1756658
time: 7.55 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1750477, upper bound: 27.1753806
time: 6.15 seconds

## BFS IS instance: IS_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -16.6616421, 12.6399870, -17.2449570, 13.0800991, -29.7417374, 29.8849430
1: -13.3036957, 11.2486629, -13.7795954, 11.6406269, -24.9443188, 25.0282593
2: -22.1320305, 7.1961861, -22.8906136, 7.4692726, -29.6013012, 30.0867996
3: -19.7262077, 8.9372892, -20.4312782, 9.2476244, -28.9738312, 29.3685608
4: -19.9779243, 12.1156988, -20.6699829, 12.5328417, -32.5107651, 32.7856789
5: -14.8639927, 12.8532848, -15.3956213, 13.2966385, -28.1606312, 28.2489052
6: -15.8787384, 13.5572023, -16.4374809, 14.0327120, -29.9114494, 29.9946804
7: -17.9093170, 13.0464077, -18.5356541, 13.5035667, -31.4128780, 31.5820580
8: -20.6123123, 11.7694168, -21.3390388, 12.1740503, -32.7863617, 33.1084557
9: -14.3378887, 16.8751411, -14.8414249, 17.4603462, -31.7982349, 31.7165642

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1813407, upper bound: 27.1813263
time: 12.64 seconds

## Relational analysis of IS_A1_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1813407, upper bound: 27.1849212
time: 5.95 seconds

## BFS IS instance: IS_A1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -16.5571899, 12.5325937, -16.9747219, 12.8585558, -29.4157410, 29.5073166
1: -13.2245874, 11.1691542, -13.5552807, 11.4467297, -24.6713181, 24.7244339
2: -22.0599403, 7.0154672, -22.5872841, 7.2338810, -29.2938213, 29.6027508
3: -19.6022568, 8.8386860, -20.1134491, 9.0787354, -28.6809921, 28.9521351
4: -19.8884125, 12.0225067, -20.3691444, 12.3228579, -32.2112694, 32.3916512
5: -14.7343597, 12.7714396, -15.1254597, 13.0903568, -27.8247128, 27.8969002
6: -15.7559643, 13.4597607, -16.1562653, 13.7969189, -29.5528831, 29.6160240
7: -17.7954788, 12.9075031, -18.2419930, 13.2500172, -31.0454941, 31.1494961
8: -20.4748306, 11.6735439, -20.9932251, 11.9673176, -32.4421463, 32.6667671
9: -14.2278252, 16.7954655, -14.5906162, 17.2083149, -31.4361401, 31.3860817

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A1_A2_A1_B1

### Relational analysis result of IS_A1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755681, upper bound: 27.1754891
time: 7.57 seconds

## Relational analysis of IS_A1_A1_A2_A1_B2

### Relational analysis result of IS_A1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754007, upper bound: 27.1753922
time: 5.79 seconds

## BFS IS instance: IS_A1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -17.8983459, 13.5634270, -17.5099297, 13.2863913, -31.1847382, 31.0733528
1: -14.3204079, 12.0806704, -13.9991465, 11.8285923, -26.1490002, 26.0798168
2: -23.7422066, 7.7699490, -23.2200527, 7.6417332, -31.3839378, 30.9899979
3: -21.2047329, 9.5790482, -20.7505951, 9.4001665, -30.6049004, 30.3296432
4: -21.4498367, 13.0025129, -20.9800701, 12.7315826, -34.1814117, 33.9825821
5: -15.9752235, 13.7889328, -15.6437025, 13.4992485, -29.4744720, 29.4326324
6: -17.0727539, 14.5649471, -16.7007713, 14.2550859, -31.3278351, 31.2657185
7: -19.2390022, 14.0072517, -18.8200912, 13.7279034, -32.9669037, 32.8273430
8: -22.1489754, 12.6298332, -21.6733875, 12.3712816, -34.5202560, 34.3032150
9: -15.4035320, 18.1156902, -15.0786171, 17.7191391, -33.1226692, 33.1943054

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_A2_A2_B1

### Relational analysis result of IS_A1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1807461, upper bound: 27.1810986
time: 7.60 seconds

## Relational analysis of IS_A1_A1_A2_A2_B2

### Relational analysis result of IS_A1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1807461, upper bound: 27.1848081
time: 6.26 seconds

## BFS IS instance: IS_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -18.5302963, 14.0358591, -16.5143299, 12.5131865, -31.0434742, 30.5501862
1: -14.8320360, 12.5049496, -13.1810818, 11.1407986, -25.9728317, 25.6860313
2: -24.5778198, 8.0257225, -21.9824982, 7.0322008, -31.6100140, 30.0082207
3: -22.0006771, 9.9191742, -19.5611877, 8.8374786, -30.8381538, 29.4803581
4: -22.2237358, 13.4625969, -19.8221054, 11.9949703, -34.2187004, 33.2846985
5: -16.5564766, 14.2775421, -14.7119904, 12.7392454, -29.2957230, 28.9895210
6: -17.6499538, 15.0752630, -15.7171307, 13.4254627, -31.0754166, 30.7923927
7: -19.9143181, 14.4862556, -17.7472324, 12.8926544, -32.8069687, 32.2334900
8: -22.9399643, 13.0549936, -20.4234905, 11.6493320, -34.5892944, 33.4784813
9: -15.9441853, 18.7583179, -14.1955929, 16.7461567, -32.6903419, 32.9539108

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1760397, upper bound: 27.1759237
time: 7.58 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1760397, upper bound: 27.1762146
time: 9.91 seconds

## BFS IS instance: IS_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -18.0096130, 13.6447926, -21.1232109, 15.9701633, -33.9797745, 34.7680016
1: -14.4097233, 12.1572094, -16.8748951, 14.1793747, -28.5890980, 29.0321007
2: -23.8983688, 7.7900643, -27.9269428, 9.1144037, -33.0127716, 35.7170029
3: -21.3761902, 9.6430635, -25.0392113, 11.2181377, -32.5943298, 34.6822739
4: -21.6068840, 13.0900593, -25.3300838, 15.3177900, -36.9246750, 38.4201431
5: -16.0871944, 13.8804216, -18.8639660, 16.2235870, -32.3107796, 32.7443886
6: -17.1501236, 14.6543360, -20.1172199, 17.1545811, -34.3047028, 34.7715454
7: -19.3570843, 14.0795536, -22.6744289, 16.4692440, -35.8263245, 36.7539825
8: -22.2924347, 12.6922112, -26.1338806, 14.8487797, -37.1412048, 38.8260880
9: -15.4963493, 18.2367496, -18.1641617, 21.3380737, -36.8344231, 36.4009094

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1756233, upper bound: 27.1756438
time: 5.61 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1756233, upper bound: 27.1756438
time: 5.50 seconds

## BFS IS instance: IS_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -20.5492687, 15.5847559, -16.0564690, 12.1652508, -32.7145157, 31.6412239
1: -16.4946156, 13.9012356, -12.8104315, 10.8346949, -27.3293114, 26.7116661
2: -27.1012230, 9.2004938, -21.4113274, 6.7826600, -33.8838844, 30.6118202
3: -24.4268398, 11.0733433, -19.0128937, 8.5859013, -33.0127373, 30.0862370
4: -24.5807343, 14.9554234, -19.2903118, 11.6628866, -36.2436218, 34.2457352
5: -18.4190598, 15.8217611, -14.2946978, 12.3931646, -30.8122215, 30.1164589
6: -19.6294575, 16.7577286, -15.2765989, 13.0558300, -32.6852875, 32.0343208
7: -22.0608292, 16.1571274, -17.2557030, 12.5219126, -34.5827408, 33.4128304
8: -25.4551849, 14.5331306, -19.8615417, 11.3286552, -36.7838402, 34.3946686
9: -17.7272854, 20.7359028, -13.7995872, 16.2980614, -34.0253448, 34.5354881

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1762777, upper bound: 27.1763240
time: 6.47 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1757944, upper bound: 27.1759536
time: 8.88 seconds

## BFS IS instance: IS_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -20.5492687, 15.5847559, -17.3427105, 13.1533728, -33.7026329, 32.9274673
1: -16.4946156, 13.9012356, -13.8588114, 11.7060719, -28.2006874, 27.7600479
2: -27.1012230, 9.2004938, -23.0169048, 7.5158501, -34.6170731, 32.2173996
3: -24.4268398, 11.0733433, -20.5492001, 9.3002548, -33.7270813, 31.6225433
4: -24.5807343, 14.9554234, -20.7849541, 12.6026697, -37.1834030, 35.7403717
5: -18.4190598, 15.8217611, -15.4839163, 13.3704815, -31.7895393, 31.3056774
6: -19.6294575, 16.7577286, -16.5320263, 14.1117773, -33.7412338, 33.2897568
7: -22.0608292, 16.1571274, -18.6401806, 13.5798330, -35.6406631, 34.7973061
8: -25.4551849, 14.5331306, -21.4609089, 12.2422466, -37.6974297, 35.9940414
9: -17.7272854, 20.7359028, -14.9257936, 17.5579929, -35.2852783, 35.6616974

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1804868, upper bound: 27.1806901
time: 8.50 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2

### Relational analysis result of IS_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1804030, upper bound: 27.1806237
time: 6.07 seconds

## BFS IS instance: IS_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -16.9045410, 12.7808332, -14.9199667, 11.3179388, -28.2224789, 27.7007999
1: -13.4809628, 11.3711405, -11.8884420, 10.0747337, -23.5556889, 23.2595825
2: -22.5425129, 7.0389085, -19.9401550, 6.2280874, -28.7705994, 26.9790630
3: -20.0171185, 8.9998903, -17.6418381, 7.9797196, -27.9968376, 26.6417274
4: -20.3014984, 12.2377472, -17.9487305, 10.8431530, -31.1446495, 30.1864777
5: -15.0266590, 13.0392399, -13.2729635, 11.5362663, -26.5629253, 26.3122025
6: -16.0778084, 13.7292395, -14.1897058, 12.1412086, -28.2190170, 27.9189415
7: -18.1569328, 13.1437254, -16.0332260, 11.6282177, -29.7851448, 29.1769524
8: -20.9052124, 11.8950510, -18.4620781, 10.5410547, -31.4462662, 30.3571281
9: -14.5050220, 17.1584511, -12.8177032, 15.1722240, -29.6772461, 29.9761543

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_A1_A1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1753916, upper bound: 27.1756343
time: 9.41 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2

### Relational analysis result of IS_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1749652, upper bound: 27.1753408
time: 6.90 seconds

## BFS IS instance: IS_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -17.3848267, 13.1599712, -16.0855713, 12.1972256, -29.5820484, 29.2455425
1: -13.8714008, 11.6979389, -12.8285780, 10.8507023, -24.7221031, 24.5265160
2: -23.1248360, 7.3628454, -21.4118958, 6.8563495, -29.9811821, 28.7747421
3: -20.5968475, 9.2772732, -19.0402031, 8.6174669, -29.2143135, 28.3174763
4: -20.8458633, 12.5986872, -19.2988510, 11.6885376, -32.5344009, 31.8975372
5: -15.4824009, 13.3998337, -14.3302727, 12.4140234, -27.8964233, 27.7301044
6: -16.5466843, 14.1255741, -15.3097763, 13.0821772, -29.6288605, 29.4353466
7: -18.6816788, 13.5562344, -17.2900238, 12.5716887, -31.2533684, 30.8462543
8: -21.4923954, 12.2432489, -19.8898220, 11.3567514, -32.8491402, 32.1330719
9: -14.9318419, 17.6217346, -13.8277130, 16.3108616, -31.2427025, 31.4494476

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_A1_A1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1801424, upper bound: 27.1801743
time: 9.57 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1798798, upper bound: 27.1800029
time: 8.67 seconds

## BFS IS instance: IS_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -18.2059612, 13.7515926, -14.9669886, 11.3527708, -29.5587311, 28.7185822
1: -14.5411015, 12.2382956, -11.9264536, 10.1057129, -24.6468143, 24.1647453
2: -24.2335949, 7.6418233, -20.0026760, 6.2474208, -30.4810123, 27.6444988
3: -21.5781860, 9.6769009, -17.6984940, 8.0040836, -29.5822697, 27.3753891
4: -21.8441315, 13.1698151, -18.0044899, 10.8763332, -32.7204666, 31.1743031
5: -16.1919441, 14.0228844, -13.3147240, 11.5719967, -27.7639351, 27.3376083
6: -17.3235798, 14.7788086, -14.2344818, 12.1789570, -29.5025330, 29.0132904
7: -19.5575638, 14.1540575, -16.0837059, 11.6642599, -31.2218246, 30.2377625
8: -22.5074005, 12.7955675, -18.5202255, 10.5733261, -33.0807266, 31.3157921
9: -15.6232777, 18.4573650, -12.8577547, 15.2194328, -30.8427105, 31.3151131

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_A1_A2_B1_B1

### Relational analysis result of IS_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1753856, upper bound: 27.1756229
time: 5.60 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2

### Relational analysis result of IS_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1749348, upper bound: 27.1753145
time: 8.80 seconds

## BFS IS instance: IS_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -18.7117481, 14.1562700, -16.1343899, 12.2335014, -30.9452419, 30.2906609
1: -14.9596910, 12.5893278, -12.8681784, 10.8829079, -25.8425980, 25.4575043
2: -24.8451157, 7.9940457, -21.4767494, 6.8767781, -31.7218933, 29.4707870
3: -22.1872139, 9.9722691, -19.0989494, 8.6428089, -30.8300228, 29.0712128
4: -22.4207592, 13.5536995, -19.3567390, 11.7230434, -34.1438026, 32.9104385
5: -16.6798592, 14.4046345, -14.3736162, 12.4511538, -29.1310120, 28.7782497
6: -17.8236198, 15.2058163, -15.3561764, 13.1214199, -30.9450302, 30.5619926
7: -20.1097717, 14.5964756, -17.3424225, 12.6092510, -32.7190247, 31.9388962
8: -23.1352463, 13.1663904, -19.9501553, 11.3904438, -34.5256882, 33.1165390
9: -16.0779514, 18.9486198, -13.8693771, 16.3598518, -32.4378052, 32.8179932

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_A1_A2_B2_B1

### Relational analysis result of IS_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1801277, upper bound: 27.1801504
time: 12.48 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2

### Relational analysis result of IS_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1798433, upper bound: 27.1799690
time: 7.66 seconds

## BFS IS instance: IS_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -19.0308037, 14.3898458, -15.3605251, 11.6440058, -30.6748047, 29.7503700
1: -15.2157574, 12.8034649, -12.2407837, 10.3658419, -25.5816002, 25.0442486
2: -25.2867603, 8.0878839, -20.4999123, 6.4575348, -31.7442932, 28.5877934
3: -22.6002884, 10.1366882, -18.1701164, 8.2199640, -30.8202515, 28.3068047
4: -22.8285828, 13.7763548, -18.4594860, 11.1615181, -33.9900970, 32.2358398
5: -16.9747734, 14.6520329, -13.6678448, 11.8661356, -28.8409081, 28.3198776
6: -18.1126823, 15.4557152, -14.6130981, 12.4925632, -30.6052456, 30.0688133
7: -20.4428635, 14.8224831, -16.5061340, 11.9800787, -32.4229431, 31.3286171
8: -23.5421352, 13.3716965, -18.9992905, 10.8467522, -34.3888855, 32.3709869
9: -16.3466911, 19.2791977, -13.1976452, 15.6015329, -31.9482155, 32.4768448

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1751426, upper bound: 27.1748738
time: 5.28 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1748413, upper bound: 27.1746337
time: 7.58 seconds

## BFS IS instance: IS_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -20.5933266, 15.5787964, -15.8255110, 12.0081711, -32.6014977, 31.4043083
1: -16.5005703, 13.8704233, -12.6188793, 10.6824255, -27.1829948, 26.4893036
2: -27.2335453, 8.9698715, -21.0567417, 6.7790518, -34.0125961, 30.0266075
3: -24.4720535, 11.0019608, -18.7289124, 8.4915686, -32.9636230, 29.7308693
4: -24.6435204, 14.9209728, -18.9854507, 11.5103035, -36.1538239, 33.9064255
5: -18.4161186, 15.8392239, -14.1044703, 12.2171669, -30.6332855, 29.9436951
6: -19.6363602, 16.7390652, -15.0659666, 12.8755884, -32.5119438, 31.8050308
7: -22.1198864, 16.1092644, -17.0129795, 12.3839703, -34.5038567, 33.1222458
8: -25.4798622, 14.4819145, -19.5680599, 11.1826534, -36.6625137, 34.0499725
9: -17.7163410, 20.8041935, -13.6105223, 16.0437489, -33.7600899, 34.4147110

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1803586, upper bound: 27.1805614
time: 16.21 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1803586, upper bound: 27.1805614
time: 9.17 seconds

## BFS IS instance: IS_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -19.0442886, 14.3999758, -16.6772232, 12.6228333, -31.6671200, 31.0771980
1: -15.2268715, 12.8125725, -13.3155117, 11.2425375, -26.4694099, 26.1280823
2: -25.3051109, 8.0935440, -22.2093849, 7.0681181, -32.3732300, 30.3029251
3: -22.6166611, 10.1439285, -19.7476883, 8.9058037, -31.5224648, 29.8916149
4: -22.8445244, 13.7859354, -20.0195923, 12.1036596, -34.9481850, 33.8055267
5: -16.9868450, 14.6624308, -14.8433647, 12.8613100, -29.8481522, 29.5057945
6: -18.1255550, 15.4666348, -15.8732100, 13.5539703, -31.6795254, 31.3398438
7: -20.4573250, 14.8329487, -17.9232979, 13.0032244, -33.4605484, 32.7562485
8: -23.5588951, 13.3811274, -20.6197433, 11.7559166, -35.3148117, 34.0008698
9: -16.3582535, 19.2927437, -14.3267956, 16.9136696, -33.2719231, 33.6195374

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1750827, upper bound: 27.1748109
time: 9.72 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1748022, upper bound: 27.1745923
time: 7.56 seconds

## BFS IS instance: IS_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -20.6061230, 15.5883236, -17.1839161, 13.0257101, -33.6318283, 32.7722397
1: -16.5111599, 13.8790236, -13.7363157, 11.5974751, -28.1086349, 27.6153393
2: -27.2507591, 8.9754000, -22.8104382, 7.4380336, -34.6887932, 31.7858391
3: -24.4875183, 11.0088472, -20.3508873, 9.2053261, -33.6928444, 31.3597317
4: -24.6585388, 14.9299126, -20.5974274, 12.4878235, -37.1463623, 35.5273361
5: -18.4274883, 15.8489885, -15.3312044, 13.2466097, -31.6740990, 31.1801872
6: -19.6484776, 16.7494011, -16.3791504, 13.9842253, -33.6326981, 33.1285515
7: -22.1334343, 16.1192341, -18.4734459, 13.4516754, -35.5851059, 34.5926819
8: -25.4956894, 14.4907598, -21.2554054, 12.1286230, -37.6243134, 35.7461624
9: -17.7273426, 20.8170185, -14.7847843, 17.3982353, -35.1255798, 35.6018028

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1802297, upper bound: 27.1804662
time: 10.75 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1802297, upper bound: 27.1804662
time: 8.17 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 20.23 seconds
IS_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1754565, upper bound: 27.1756658
IS_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1750477, upper bound: 27.1753806
IS_A1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1813407, upper bound: 27.1813263
IS_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1813407, upper bound: 27.1849212
IS_A1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1755681, upper bound: 27.1754891
IS_A1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1754007, upper bound: 27.1753922
IS_A1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1807461, upper bound: 27.1810986
IS_A1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1807461, upper bound: 27.1848081
IS_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1760397, upper bound: 27.1759237
IS_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1760397, upper bound: 27.1762146
IS_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1756233, upper bound: 27.1756438
IS_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1756233, upper bound: 27.1756438
IS_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1762777, upper bound: 27.1763240
IS_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1757944, upper bound: 27.1759536
IS_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1804868, upper bound: 27.1806901
IS_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1804030, upper bound: 27.1806237
IS_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1753916, upper bound: 27.1756343
IS_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1749652, upper bound: 27.1753408
IS_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1801424, upper bound: 27.1801743
IS_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1798798, upper bound: 27.1800029
IS_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1753856, upper bound: 27.1756229
IS_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1749348, upper bound: 27.1753145
IS_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1801277, upper bound: 27.1801504
IS_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1798433, upper bound: 27.1799690
IS_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1751426, upper bound: 27.1748738
IS_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1748413, upper bound: 27.1746337
IS_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1803586, upper bound: 27.1805614
IS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1803586, upper bound: 27.1805614
IS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1750827, upper bound: 27.1748109
IS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1748022, upper bound: 27.1745923
IS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1802297, upper bound: 27.1804662
IS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 2, lower bound: -27.1802297, upper bound: 27.1804662

## BFS IS instance: IS_A1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -15.8627901, 12.0210533, -15.4369173, 11.7041492, -27.5669403, 27.4579697
1: -12.6517467, 10.7041340, -12.3092957, 10.4248352, -23.0765820, 23.0134296
2: -21.1435623, 6.7080040, -20.5989380, 6.5115275, -27.6550884, 27.3069420
3: -18.7713509, 8.4856968, -18.2680016, 8.2613144, -27.0326595, 26.7536983
4: -19.0540295, 11.5247459, -18.5573807, 11.2225266, -30.2765560, 30.0821190
5: -14.1204872, 12.2444744, -13.7427607, 11.9230881, -26.0435734, 25.9872360
6: -15.0945244, 12.8979378, -14.6851234, 12.5579348, -27.6524582, 27.5830593
7: -17.0453911, 12.3760147, -16.5898132, 12.0424938, -29.0878849, 28.9658241
8: -19.6173096, 11.1956482, -19.0972023, 10.9014206, -30.5187225, 30.2928505
9: -13.6326599, 16.0984592, -13.2682924, 15.6780119, -29.3106689, 29.3667526

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752179, upper bound: 27.1752754
time: 5.00 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752179, upper bound: 27.1752754
time: 22.36 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -15.3598080, 11.6463718, -19.6942291, 14.8504496, -30.2102585, 31.3406010
1: -12.2451191, 10.3712835, -15.7117615, 13.2134571, -25.4585724, 26.0830460
2: -20.4850788, 6.4890943, -26.1232109, 8.3467255, -28.8318043, 32.6123009
3: -18.1658497, 8.2228813, -23.3408432, 10.4397087, -28.6055527, 31.5637245
4: -18.4583244, 11.1676693, -23.6501293, 14.2545929, -32.7129173, 34.8177948
5: -13.6715374, 11.8628559, -17.5153008, 15.1284609, -28.7999973, 29.3781509
6: -14.6139421, 12.4932184, -18.7214794, 15.9738522, -30.5877934, 31.2146950
7: -16.5052204, 11.9871616, -21.1550217, 15.3021164, -31.8073349, 33.1421814
8: -18.9961090, 10.8491325, -24.3332367, 13.8242874, -32.8203888, 35.1823692
9: -13.2009964, 15.5955534, -16.9137821, 19.9164352, -33.1174316, 32.5093346

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1748890, upper bound: 27.1750998
time: 6.18 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1748890, upper bound: 27.1753806
time: 7.32 seconds

## BFS IS instance: IS_A1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -15.2140265, 11.5365515, -17.2449570, 13.0800991, -28.2941246, 28.7815075
1: -12.1283741, 10.2741899, -13.7795954, 11.6406269, -23.7689991, 24.0537853
2: -20.3167534, 6.3859243, -22.8906136, 7.4692726, -27.7860203, 29.2765388
3: -17.9967804, 8.1379576, -20.4312782, 9.2476244, -27.2444038, 28.5692329
4: -18.2974281, 11.0589390, -20.6699829, 12.5328417, -30.8302689, 31.7289200
5: -13.5382442, 11.7565327, -15.3956213, 13.2966385, -26.8348808, 27.1521530
6: -14.4727058, 12.3783264, -16.4374809, 14.0327120, -28.5054169, 28.8158016
7: -16.3494530, 11.8629913, -18.5356541, 13.5035667, -29.8530178, 30.3986435
8: -18.8234997, 10.7458096, -21.3390388, 12.1740503, -30.9975510, 32.0848465
9: -13.0748196, 15.4601221, -14.8414249, 17.4603462, -30.5351658, 30.3015480

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752179, upper bound: 27.1755514
time: 6.78 seconds

## Relational analysis of IS_A1_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1748890, upper bound: 27.1754735
time: 8.97 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 17.02 seconds
IS_A1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 17.02
Output dim: 2, lower bound: -27.1752179, upper bound: 27.1752754
IS_A1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 17.02
Output dim: 2, lower bound: -27.1752179, upper bound: 27.1752754
IS_A1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 17.02
Output dim: 2, lower bound: -27.1748890, upper bound: 27.1750998
IS_A1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 17.02
Output dim: 2, lower bound: -27.1748890, upper bound: 27.1753806
IS_A1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 17.02
Output dim: 2, lower bound: -27.1752179, upper bound: 27.1755514
IS_A1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 17.02
Output dim: 2, lower bound: -27.1748890, upper bound: 27.1754735
IS_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1813407, upper bound: 27.1849212
IS_A1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1755681, upper bound: 27.1754891
IS_A1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1754007, upper bound: 27.1753922
IS_A1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1807461, upper bound: 27.1810986
IS_A1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1807461, upper bound: 27.1848081
IS_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1760397, upper bound: 27.1759237
IS_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1760397, upper bound: 27.1762146
IS_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1756233, upper bound: 27.1756438
IS_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1756233, upper bound: 27.1756438
IS_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1762777, upper bound: 27.1763240
IS_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1757944, upper bound: 27.1759536
IS_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1804868, upper bound: 27.1806901
IS_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1804030, upper bound: 27.1806237
IS_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1753916, upper bound: 27.1756343
IS_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1749652, upper bound: 27.1753408
IS_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1801424, upper bound: 27.1801743
IS_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1798798, upper bound: 27.1800029
IS_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1753856, upper bound: 27.1756229
IS_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1749348, upper bound: 27.1753145
IS_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1801277, upper bound: 27.1801504
IS_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1798433, upper bound: 27.1799690
IS_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1751426, upper bound: 27.1748738
IS_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1748413, upper bound: 27.1746337
IS_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1803586, upper bound: 27.1805614
IS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1803586, upper bound: 27.1805614
IS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1750827, upper bound: 27.1748109
IS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1748022, upper bound: 27.1745923
IS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1802297, upper bound: 27.1804662
IS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -27.1802297, upper bound: 27.1804662
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=36.862388610839844
rel_dist={2: [-27.200440708568877, 27.200440707169562]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1818.06 seconds
