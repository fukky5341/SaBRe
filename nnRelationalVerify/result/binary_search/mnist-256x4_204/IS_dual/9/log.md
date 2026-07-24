## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 33.8747337875
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899)
1: (-17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774)
2: (-22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451)
3: (-28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570)
4: (-25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998)
5: (-21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947)
6: (-20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817)
7: (-24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103)
8: (-28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781)
9: (-23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068)

## BASE Result
execution time: IAR + LP analysis = 1.23 + 8.38 = 9.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -33.8800858, upper bound: 33.8800858


# Binary Search by BASE starts (time budget: 2690.39 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=41.571006774902344
rel_dist={9: [-33.88006301494167, 33.88006301377598]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=41.571006774902344
rel_dist={9: [-33.88001792224934, 33.88001792519913]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=41.571006774902344
rel_dist={9: [-33.879905218442005, 33.87990521540584]}

## Binary Search Result
Binary search time: 24.10 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2666.28 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8772620, upper bound: 33.8745820
time: 6.25 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8772620, upper bound: 33.8795380
time: 7.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.14 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 14.14
Output dim: 9, lower bound: -33.8772620, upper bound: 33.8745820
IS_B2, status: Status.UNKNOWN, split count: 1, time: 14.14
Output dim: 9, lower bound: -33.8772620, upper bound: 33.8795380

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -22.7433853, 19.1949692, -21.7221832, 18.3904572, -41.1338425, 40.9171524
1: -16.8248177, 16.0234566, -16.0281563, 15.3024387, -32.1272583, 32.0516129
2: -22.3099117, 17.4886856, -21.2368584, 16.7477531, -39.0576630, 38.7255440
3: -27.9693069, 13.8722591, -26.7382374, 13.2450066, -41.2143135, 40.6104965
4: -25.0436821, 17.3254089, -23.9092598, 16.5171890, -41.5608673, 41.2346649
5: -21.1934319, 17.1861687, -20.2175159, 16.3721180, -37.5655518, 37.4036865
6: -20.2833252, 20.2983017, -19.3129272, 19.4132710, -39.6965866, 39.6112289
7: -23.6803932, 19.2537975, -22.6419182, 18.4087086, -42.0890961, 41.8957138
8: -27.9991493, 17.9363632, -26.7138176, 17.1098518, -45.1090012, 44.6501732
9: -23.4206829, 17.1895885, -22.4509315, 16.2217712, -39.6424522, 39.6405182

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8723647, upper bound: 33.8705758
time: 12.76 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8747123, upper bound: 33.8712064
time: 14.41 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -23.2670193, 19.6152725, -22.6191711, 19.0941982, -42.3612175, 42.2344398
1: -17.2405739, 16.3867092, -16.7274170, 15.9382935, -33.1788597, 33.1141281
2: -22.8407764, 17.8757706, -22.1822586, 17.3967056, -40.2374802, 40.0580292
3: -28.6123161, 14.1993437, -27.8153362, 13.7984829, -42.4107971, 42.0146790
4: -25.6187172, 17.7315826, -24.9088879, 17.2293472, -42.8480644, 42.6404724
5: -21.6950798, 17.5882168, -21.0763359, 17.0936165, -38.7886963, 38.6645508
6: -20.7598629, 20.7582188, -20.1703262, 20.1893749, -40.9492378, 40.9285431
7: -24.2113094, 19.6866989, -23.5521469, 19.1513710, -43.3626785, 43.2388458
8: -28.6491680, 18.3628082, -27.8460331, 17.8391724, -46.4883385, 46.2088394
9: -23.9259472, 17.6450577, -23.3010559, 17.0844669, -41.0104141, 40.9461136

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8745820, upper bound: 33.8772620
time: 4.91 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8745820, upper bound: 33.8795380
time: 4.09 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 10.40 seconds
IS_B1_A1, status: Status.VERIFIED, split count: 2, time: 10.40
Output dim: 9, lower bound: -33.8723647, upper bound: 33.8705758
IS_B1_A2, status: Status.VERIFIED, split count: 2, time: 10.40
Output dim: 9, lower bound: -33.8747123, upper bound: 33.8712064
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 10.40
Output dim: 9, lower bound: -33.8745820, upper bound: 33.8772620
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 10.40
Output dim: 9, lower bound: -33.8745820, upper bound: 33.8795380

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -21.7221832, 18.3904572, -22.6191711, 19.0941982, -40.8163795, 41.0096283
1: -16.0281563, 15.3024387, -16.7274170, 15.9382935, -31.9664497, 32.0298538
2: -21.2368584, 16.7477531, -22.1822586, 17.3967056, -38.6335640, 38.9300079
3: -26.7382374, 13.2450066, -27.8153362, 13.7984829, -40.5367203, 41.0603409
4: -23.9092598, 16.5171890, -24.9088879, 17.2293472, -41.1386032, 41.4260750
5: -20.2175159, 16.3721180, -21.0763359, 17.0936165, -37.3111305, 37.4484520
6: -19.3129272, 19.4132710, -20.1703262, 20.1893749, -39.5023003, 39.5835953
7: -22.6419182, 18.4087086, -23.5521469, 19.1513710, -41.7932892, 41.9608536
8: -26.7138176, 17.1098518, -27.8460331, 17.8391724, -44.5529900, 44.9558868
9: -22.4509315, 16.2217712, -23.3010559, 17.0844669, -39.5354004, 39.5228271

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8701515, upper bound: 33.8723647
time: 5.23 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8705932, upper bound: 33.8747123
time: 10.12 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -22.6191711, 19.0941982, -22.6191711, 19.0941982, -41.7133675, 41.7133675
1: -16.7274170, 15.9382935, -16.7274170, 15.9382935, -32.6657066, 32.6657066
2: -22.1822586, 17.3967056, -22.1822586, 17.3967056, -39.5789604, 39.5789642
3: -27.8153362, 13.7984829, -27.8153362, 13.7984829, -41.6138191, 41.6138191
4: -24.9088879, 17.2293472, -24.9088879, 17.2293472, -42.1382294, 42.1382294
5: -21.0763359, 17.0936165, -21.0763359, 17.0936165, -38.1699524, 38.1699524
6: -20.1703262, 20.1893749, -20.1703262, 20.1893749, -40.3597031, 40.3597031
7: -23.5521469, 19.1513710, -23.5521469, 19.1513710, -42.7035179, 42.7035179
8: -27.8460331, 17.8391724, -27.8460331, 17.8391724, -45.6852036, 45.6852036
9: -23.3010559, 17.0844669, -23.3010559, 17.0844669, -40.3855209, 40.3855209

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8662751, upper bound: 33.8751575
time: 6.26 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8657557, upper bound: 33.8727795
time: 3.34 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 10.88 seconds
IS_B2_A1_B1, status: Status.VERIFIED, split count: 3, time: 10.88
Output dim: 9, lower bound: -33.8701515, upper bound: 33.8723647
IS_B2_A1_B2, status: Status.VERIFIED, split count: 3, time: 10.88
Output dim: 9, lower bound: -33.8705932, upper bound: 33.8747123
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 10.88
Output dim: 9, lower bound: -33.8662751, upper bound: 33.8751575
IS_B2_A2_B2, status: Status.VERIFIED, split count: 3, time: 10.88
Output dim: 9, lower bound: -33.8657557, upper bound: 33.8727795

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -22.6191711, 19.0941982, -17.8729916, 15.1854610, -37.8046341, 36.9671898
1: -16.7274170, 15.9382935, -13.1922665, 12.6481838, -29.3756008, 29.1305561
2: -22.1822586, 17.3967056, -17.4508247, 13.8467379, -36.0289955, 34.8475304
3: -27.8153362, 13.7984829, -21.8948402, 10.9931774, -38.8085136, 35.6933212
4: -24.9088879, 17.2293472, -19.6907043, 13.6641045, -38.5729904, 36.9200516
5: -21.0763359, 17.0936165, -16.6374321, 13.5545778, -34.6309128, 33.7310486
6: -20.1703262, 20.1893749, -15.9128904, 16.0212212, -36.1915474, 36.1022644
7: -23.5521469, 19.1513710, -18.6249924, 15.2347565, -38.7869034, 37.7763634
8: -27.8460331, 17.8391724, -22.0153484, 14.1560154, -42.0020447, 39.8545227
9: -23.3010559, 17.0844669, -18.5149460, 13.5278664, -36.8289223, 35.5994110

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8678107, upper bound: 33.8715201
time: 6.21 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8699468, upper bound: 33.8723232
time: 3.91 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 11.40 seconds
IS_B2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 11.40
Output dim: 9, lower bound: -33.8678107, upper bound: 33.8715201
IS_B2_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 11.40
Output dim: 9, lower bound: -33.8699468, upper bound: 33.8723232
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=41.571006774902344
rel_dist={9: [-33.88006301494167, 33.88006301377598]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8747448, upper bound: 33.8780329
time: 5.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8795485, upper bound: 33.8795486
time: 4.77 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.47 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.47
Output dim: 9, lower bound: -33.8747448, upper bound: 33.8780329
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.47
Output dim: 9, lower bound: -33.8795485, upper bound: 33.8795486

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -21.7221832, 18.3904572, -23.1402931, 19.5140305, -41.2362137, 41.5307503
1: -16.0281563, 15.3024387, -17.1399727, 16.2987022, -32.3268585, 32.4424095
2: -21.2368584, 16.7477531, -22.7121677, 17.7825737, -39.0194321, 39.4599228
3: -26.7382374, 13.2450066, -28.4567337, 14.1207161, -40.8589554, 41.7017365
4: -23.9092598, 16.5171890, -25.4800873, 17.6332970, -41.5425529, 41.9972649
5: -20.2175159, 16.3721180, -21.5738716, 17.4913654, -37.7088814, 37.9459839
6: -19.3129272, 19.4132710, -20.6445389, 20.6473312, -39.9602509, 40.0578079
7: -22.6419182, 18.4087086, -24.0833702, 19.5820198, -42.2239380, 42.4920731
8: -26.7138176, 17.1098518, -28.4922504, 18.2600346, -44.9738541, 45.6021042
9: -22.4509315, 16.2217712, -23.8042488, 17.5348415, -39.9857712, 40.0260201

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8708984, upper bound: 33.8732195
time: 5.13 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8714229, upper bound: 33.8756488
time: 6.45 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -22.6191711, 19.0941982, -23.2670193, 19.6152725, -42.2344398, 42.3612137
1: -16.7274170, 15.9382935, -17.2405739, 16.3867092, -33.1141281, 33.1788597
2: -22.1822586, 17.3967056, -22.8407764, 17.8757706, -40.0580292, 40.2374802
3: -27.8153362, 13.7984829, -28.6123161, 14.1993437, -42.0146790, 42.4107971
4: -24.9088879, 17.2293472, -25.6187172, 17.7315826, -42.6404724, 42.8480644
5: -21.0763359, 17.0936165, -21.6950798, 17.5882168, -38.6645508, 38.7886963
6: -20.1703262, 20.1893749, -20.7598629, 20.7582188, -40.9285431, 40.9492378
7: -23.5521469, 19.1513710, -24.2113094, 19.6866989, -43.2388458, 43.3626785
8: -27.8460331, 17.8391724, -28.6491680, 18.3628082, -46.2088394, 46.4883423
9: -23.3010559, 17.0844669, -23.9259472, 17.6450577, -40.9461136, 41.0104141

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8733405, upper bound: 33.8756247
time: 3.85 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8727881, upper bound: 33.8727881
time: 3.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 8.96 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 8.96
Output dim: 9, lower bound: -33.8708984, upper bound: 33.8732195
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 8.96
Output dim: 9, lower bound: -33.8714229, upper bound: 33.8756488
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 8.96
Output dim: 9, lower bound: -33.8733405, upper bound: 33.8756247
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 8.96
Output dim: 9, lower bound: -33.8727881, upper bound: 33.8727881

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -21.7221832, 18.3904572, -22.5403709, 19.0462093, -40.7683945, 40.9308281
1: -16.0281563, 15.3024387, -16.6255379, 15.8437119, -31.8718681, 31.9279766
2: -21.2368584, 16.7477531, -21.9558754, 17.3283558, -38.5652084, 38.7036285
3: -26.7382374, 13.2450066, -27.8130798, 13.7108707, -40.4491081, 41.0580826
4: -23.9092598, 16.5171890, -24.7966805, 17.0326748, -40.9419327, 41.3138618
5: -20.2175159, 16.3721180, -20.9599571, 16.9263573, -37.1438751, 37.3320732
6: -19.3129272, 19.4132710, -19.9837246, 20.1340866, -39.4470139, 39.3969955
7: -22.6419182, 18.4087086, -23.4388485, 19.0014572, -41.6433754, 41.8475571
8: -26.7138176, 17.1098518, -27.6868439, 17.7255020, -44.4393196, 44.7966957
9: -22.4509315, 16.2217712, -23.2217331, 16.7311935, -39.1821213, 39.4434929

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8665248, upper bound: 33.8681579
time: 3.84 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8620960, upper bound: 33.8671326
time: 26.43 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -22.6191711, 19.0941982, -18.4010162, 15.6114140, -38.2305832, 37.4952164
1: -16.7274170, 15.9382935, -13.5919094, 13.0152292, -29.7426453, 29.5301933
2: -22.1822586, 17.3967056, -17.9768524, 14.2358227, -36.4180756, 35.3735580
3: -27.8153362, 13.7984829, -22.5516376, 11.3189268, -39.1342621, 36.3501167
4: -24.9088879, 17.2293472, -20.2675476, 14.0601978, -38.9690857, 37.4968910
5: -21.0763359, 17.0936165, -17.1344261, 13.9622078, -35.0385437, 34.2280426
6: -20.1703262, 20.1893749, -16.3914318, 16.4827347, -36.6530609, 36.5808029
7: -23.5521469, 19.1513710, -19.1605721, 15.6680775, -39.2202225, 38.3119431
8: -27.8460331, 17.8391724, -22.6600533, 14.5711079, -42.4171333, 40.4992256
9: -23.3010559, 17.0844669, -19.0300560, 13.9656067, -37.2666626, 36.1145248

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8681080, upper bound: 33.8723538
time: 6.24 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8701515, upper bound: 33.8728878
time: 4.14 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 11.66 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 11.66
Output dim: 9, lower bound: -33.8665248, upper bound: 33.8681579
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 11.66
Output dim: 9, lower bound: -33.8620960, upper bound: 33.8671326
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 11.66
Output dim: 9, lower bound: -33.8681080, upper bound: 33.8723538
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 11.66
Output dim: 9, lower bound: -33.8701515, upper bound: 33.8728878
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=41.571006774902344
rel_dist={9: [-33.88007495180708, 33.88007494630283]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8748509, upper bound: 33.8783311
time: 3.12 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8795555, upper bound: 33.8795556
time: 3.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 6.89 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 6.89
Output dim: 9, lower bound: -33.8748509, upper bound: 33.8783311
IS_A2, status: Status.UNKNOWN, split count: 1, time: 6.89
Output dim: 9, lower bound: -33.8795555, upper bound: 33.8795556

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -21.7221832, 18.3904572, -23.2670193, 19.6152725, -41.3374557, 41.6574783
1: -16.0281563, 15.3024387, -17.2405739, 16.3867092, -32.4148636, 32.5430107
2: -21.2368584, 16.7477531, -22.8407764, 17.8757706, -39.1126289, 39.5885315
3: -26.7382374, 13.2450066, -28.6123161, 14.1993437, -40.9375801, 41.8573151
4: -23.9092598, 16.5171890, -25.6187172, 17.7315826, -41.6408424, 42.1359024
5: -20.2175159, 16.3721180, -21.6950798, 17.5882168, -37.8057289, 38.0671959
6: -19.3129272, 19.4132710, -20.7598629, 20.7582188, -40.0711441, 40.1731339
7: -22.6419182, 18.4087086, -24.2113094, 19.6866989, -42.3286171, 42.6200180
8: -26.7138176, 17.1098518, -28.6491680, 18.3628082, -45.0766258, 45.7590179
9: -22.4509315, 16.2217712, -23.9259472, 17.6450577, -40.0959892, 40.1477203

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8710671, upper bound: 33.8735803
time: 5.89 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8715604, upper bound: 33.8760017
time: 5.48 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -22.6191711, 19.0941982, -23.2670193, 19.6152725, -42.2344398, 42.3612137
1: -16.7274170, 15.9382935, -17.2405739, 16.3867092, -33.1141281, 33.1788597
2: -22.1822586, 17.3967056, -22.8407764, 17.8757706, -40.0580292, 40.2374802
3: -27.8153362, 13.7984829, -28.6123161, 14.1993437, -42.0146790, 42.4107971
4: -24.9088879, 17.2293472, -25.6187172, 17.7315826, -42.6404724, 42.8480644
5: -21.0763359, 17.0936165, -21.6950798, 17.5882168, -38.6645508, 38.7886963
6: -20.1703262, 20.1893749, -20.7598629, 20.7582188, -40.9285431, 40.9492378
7: -23.5521469, 19.1513710, -24.2113094, 19.6866989, -43.2388458, 43.3626785
8: -27.8460331, 17.8391724, -28.6491680, 18.3628082, -46.2088394, 46.4883423
9: -23.3010559, 17.0844669, -23.9259472, 17.6450577, -40.9461136, 41.0104141

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8734476, upper bound: 33.8758383
time: 3.09 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8727938, upper bound: 33.8727938
time: 2.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 7.05 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 7.05
Output dim: 9, lower bound: -33.8710671, upper bound: 33.8735803
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 7.05
Output dim: 9, lower bound: -33.8715604, upper bound: 33.8760017
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 7.05
Output dim: 9, lower bound: -33.8734476, upper bound: 33.8758383
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 7.05
Output dim: 9, lower bound: -33.8727938, upper bound: 33.8727938

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -21.7221832, 18.3904572, -22.6394768, 19.1257858, -40.8479652, 41.0299263
1: -16.0281563, 15.3024387, -16.7014236, 15.9111223, -31.9392776, 32.0038567
2: -21.2368584, 16.7477531, -22.0544395, 17.4008713, -38.6377296, 38.8021927
3: -26.7382374, 13.2450066, -27.9338169, 13.7699480, -40.5081863, 41.1788216
4: -23.9092598, 16.5171890, -24.9047699, 17.1077995, -41.0170593, 41.4219513
5: -20.2175159, 16.3721180, -21.0542793, 17.0026321, -37.2201462, 37.4263992
6: -19.3129272, 19.4132710, -20.0734768, 20.2198086, -39.5327301, 39.4867477
7: -22.6419182, 18.4087086, -23.5392380, 19.0824890, -41.7244072, 41.9479446
8: -26.7138176, 17.1098518, -27.8070374, 17.8020134, -44.5158310, 44.9168892
9: -22.4509315, 16.2217712, -23.3164005, 16.8133984, -39.2643280, 39.5381699

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8669603, upper bound: 33.8686600
time: 4.89 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8622595, upper bound: 33.8675215
time: 4.09 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -22.6191711, 19.0941982, -18.4010162, 15.6114140, -38.2305832, 37.4952164
1: -16.7274170, 15.9382935, -13.5919094, 13.0152292, -29.7426453, 29.5301933
2: -22.1822586, 17.3967056, -17.9768524, 14.2358227, -36.4180756, 35.3735580
3: -27.8153362, 13.7984829, -22.5516376, 11.3189268, -39.1342621, 36.3501167
4: -24.9088879, 17.2293472, -20.2675476, 14.0601978, -38.9690857, 37.4968910
5: -21.0763359, 17.0936165, -17.1344261, 13.9622078, -35.0385437, 34.2280426
6: -20.1703262, 20.1893749, -16.3914318, 16.4827347, -36.6530609, 36.5808029
7: -23.5521469, 19.1513710, -19.1605721, 15.6680775, -39.2202225, 38.3119431
8: -27.8460331, 17.8391724, -22.6600533, 14.5711079, -42.4171333, 40.4992256
9: -23.3010559, 17.0844669, -19.0300560, 13.9656067, -37.2666626, 36.1145248

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8682811, upper bound: 33.8726781
time: 19.31 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8702811, upper bound: 33.8731232
time: 11.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 32.40 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 32.40
Output dim: 9, lower bound: -33.8669603, upper bound: 33.8686600
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 32.40
Output dim: 9, lower bound: -33.8622595, upper bound: 33.8675215
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 32.40
Output dim: 9, lower bound: -33.8682811, upper bound: 33.8726781
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 32.40
Output dim: 9, lower bound: -33.8702811, upper bound: 33.8731232
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=41.571006774902344
rel_dist={9: [-33.88008219383614, 33.88008219146937]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8749025, upper bound: 33.8784395
time: 11.08 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8795591, upper bound: 33.8795590
time: 2.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.95 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.95
Output dim: 9, lower bound: -33.8749025, upper bound: 33.8784395
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.95
Output dim: 9, lower bound: -33.8795591, upper bound: 33.8795590

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -21.7221832, 18.3904572, -23.2670193, 19.6152725, -41.3374557, 41.6574783
1: -16.0281563, 15.3024387, -17.2405739, 16.3867092, -32.4148636, 32.5430107
2: -21.2368584, 16.7477531, -22.8407764, 17.8757706, -39.1126289, 39.5885315
3: -26.7382374, 13.2450066, -28.6123161, 14.1993437, -40.9375801, 41.8573151
4: -23.9092598, 16.5171890, -25.6187172, 17.7315826, -41.6408424, 42.1359024
5: -20.2175159, 16.3721180, -21.6950798, 17.5882168, -37.8057289, 38.0671959
6: -19.3129272, 19.4132710, -20.7598629, 20.7582188, -40.0711441, 40.1731339
7: -22.6419182, 18.4087086, -24.2113094, 19.6866989, -42.3286171, 42.6200180
8: -26.7138176, 17.1098518, -28.6491680, 18.3628082, -45.0766258, 45.7590179
9: -22.4509315, 16.2217712, -23.9259472, 17.6450577, -40.0959892, 40.1477203

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8711475, upper bound: 33.8737149
time: 3.70 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8716281, upper bound: 33.8761292
time: 3.29 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -22.6191711, 19.0941982, -23.2670193, 19.6152725, -42.2344398, 42.3612137
1: -16.7274170, 15.9382935, -17.2405739, 16.3867092, -33.1141281, 33.1788597
2: -22.1822586, 17.3967056, -22.8407764, 17.8757706, -40.0580292, 40.2374802
3: -27.8153362, 13.7984829, -28.6123161, 14.1993437, -42.0146790, 42.4107971
4: -24.9088879, 17.2293472, -25.6187172, 17.7315826, -42.6404724, 42.8480644
5: -21.0763359, 17.0936165, -21.6950798, 17.5882168, -38.6645508, 38.7886963
6: -20.1703262, 20.1893749, -20.7598629, 20.7582188, -40.9285431, 40.9492378
7: -23.5521469, 19.1513710, -24.2113094, 19.6866989, -43.2388458, 43.3626785
8: -27.8460331, 17.8391724, -28.6491680, 18.3628082, -46.2088394, 46.4883423
9: -23.3010559, 17.0844669, -23.9259472, 17.6450577, -40.9461136, 41.0104141

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8734958, upper bound: 33.8759172
time: 3.27 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8727966, upper bound: 33.8727966
time: 4.24 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 9.06 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 9.06
Output dim: 9, lower bound: -33.8711475, upper bound: 33.8737149
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 9.06
Output dim: 9, lower bound: -33.8716281, upper bound: 33.8761292
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 9.06
Output dim: 9, lower bound: -33.8734958, upper bound: 33.8759172
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 9.06
Output dim: 9, lower bound: -33.8727966, upper bound: 33.8727966

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -21.7221832, 18.3904572, -22.6394768, 19.1257858, -40.8479652, 41.0299263
1: -16.0281563, 15.3024387, -16.7014236, 15.9111223, -31.9392776, 32.0038567
2: -21.2368584, 16.7477531, -22.0544395, 17.4008713, -38.6377296, 38.8021927
3: -26.7382374, 13.2450066, -27.9338169, 13.7699480, -40.5081863, 41.1788216
4: -23.9092598, 16.5171890, -24.9047699, 17.1077995, -41.0170593, 41.4219513
5: -20.2175159, 16.3721180, -21.0542793, 17.0026321, -37.2201462, 37.4263992
6: -19.3129272, 19.4132710, -20.0734768, 20.2198086, -39.5327301, 39.4867477
7: -22.6419182, 18.4087086, -23.5392380, 19.0824890, -41.7244072, 41.9479446
8: -26.7138176, 17.1098518, -27.8070374, 17.8020134, -44.5158310, 44.9168892
9: -22.4509315, 16.2217712, -23.3164005, 16.8133984, -39.2643280, 39.5381699

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8671307, upper bound: 33.8688654
time: 3.41 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8623383, upper bound: 33.8676763
time: 3.28 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -22.6191711, 19.0941982, -18.4010162, 15.6114140, -38.2305832, 37.4952164
1: -16.7274170, 15.9382935, -13.5919094, 13.0152292, -29.7426453, 29.5301933
2: -22.1822586, 17.3967056, -17.9768524, 14.2358227, -36.4180756, 35.3735580
3: -27.8153362, 13.7984829, -22.5516376, 11.3189268, -39.1342621, 36.3501167
4: -24.9088879, 17.2293472, -20.2675476, 14.0601978, -38.9690857, 37.4968910
5: -21.0763359, 17.0936165, -17.1344261, 13.9622078, -35.0385437, 34.2280426
6: -20.1703262, 20.1893749, -16.3914318, 16.4827347, -36.6530609, 36.5808029
7: -23.5521469, 19.1513710, -19.1605721, 15.6680775, -39.2202225, 38.3119431
8: -27.8460331, 17.8391724, -22.6600533, 14.5711079, -42.4171333, 40.4992256
9: -23.3010559, 17.0844669, -19.0300560, 13.9656067, -37.2666626, 36.1145248

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8683634, upper bound: 33.8728009
time: 2.60 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8703437, upper bound: 33.8732136
time: 4.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 8.72 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 8.72
Output dim: 9, lower bound: -33.8671307, upper bound: 33.8688654
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 8.72
Output dim: 9, lower bound: -33.8623383, upper bound: 33.8676763
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 8.72
Output dim: 9, lower bound: -33.8683634, upper bound: 33.8728009
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 8.72
Output dim: 9, lower bound: -33.8703437, upper bound: 33.8732136
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=41.571006774902344
rel_dist={9: [-33.88008576028902, 33.88008575296402]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 319.49 seconds
