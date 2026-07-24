## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.0301740019999999
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.1949067, -1.9940324, -6.1949067, -1.9940324, -4.2008743, 4.2008743)
1: (-12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833)
2: (-5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.3852363, 3.3852363)
3: (-5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837)
4: (-11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.9559875, 3.9559875)
5: (-6.2900958, -3.0817809, -6.2900958, -3.0817809, -3.2083149, 3.2083149)
6: (-12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.7321682, 3.7321682)
7: (-8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888)
8: (7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153)
9: (-6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.5451798, 3.5451798)

## BASE Result
execution time: IAR + LP analysis = 14.26 + 39.89 = 54.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -1.7692351, upper bound: 1.7692330


# Binary Search by BASE starts (time budget: 3545.85 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.321315288543701
rel_dist={8: [-1.3149698422491287, 1.314969004521421]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.225489616394043
rel_dist={8: [-1.040580051898944, 1.0405791313826693]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.099544048309326
rel_dist={8: [-0.8121649022104638, 0.8121633074202457]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.1625168323516846
rel_dist={8: [-0.9337397585648102, 0.9337386492559308]}

## Binary Search Result
Binary search time: 210.29 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3335.56 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976653, upper bound: 1.3998450
time: 5.50 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3998453, upper bound: 1.3998452
time: 14.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 20.06 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 20.06
Output dim: 8, lower bound: -1.3976653, upper bound: 1.3998450
IS_B2, status: Status.UNKNOWN, split count: 1, time: 20.06
Output dim: 8, lower bound: -1.3998453, upper bound: 1.3998452

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -6.1931753, -1.9940886, -6.1727219, -1.9947731, -3.7086563, 3.6892200
1: -12.2333479, -8.9912958, -12.2193813, -8.9918165, -3.2321544, 3.2187595
2: -5.6192346, -2.2357464, -5.6021371, -2.2392807, -3.0963535, 3.0821605
3: -5.3707466, -1.5044949, -5.3690162, -1.5078959, -3.8628507, 3.8645213
4: -11.5234308, -7.5686359, -11.5189686, -7.5783043, -3.7529030, 3.7583394
5: -6.2899170, -3.0820436, -6.2878017, -3.0851064, -2.9587436, 2.9607244
6: -12.4277344, -8.6958666, -12.4260111, -8.6976948, -3.3578720, 3.3589287
7: -8.1699448, -4.6728067, -8.1650391, -4.6791401, -3.4908047, 3.4922323
8: 7.7390380, 10.0597591, 7.7414150, 10.0551043, -2.3160663, 2.3183441
9: -6.3477511, -2.8030176, -6.3445268, -2.8050718, -3.2628856, 3.2617588

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976653, upper bound: 1.3976648
time: 4.74 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976653, upper bound: 1.3998449
time: 4.85 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -6.1948948, -1.9940333, -6.2037158, -1.9477437, -3.7575717, 3.7182484
1: -12.2345238, -8.9912510, -12.2469511, -8.9744186, -3.2515707, 3.2470250
2: -5.6206751, -2.2354498, -5.6229401, -2.1984048, -3.1354294, 3.1025524
3: -5.3708897, -1.5042095, -5.3863273, -1.5006928, -3.8701968, 3.8821177
4: -11.5238028, -7.5678163, -11.5466595, -7.5620070, -3.7695503, 3.7863002
5: -6.2900934, -3.0817823, -6.3025022, -3.0794778, -2.9786091, 2.9759755
6: -12.4278784, -8.6957140, -12.4390497, -8.6935787, -3.3809423, 3.3713551
7: -8.1703548, -4.6722727, -8.1877203, -4.6719475, -3.4984074, 3.5154476
8: 7.7388391, 10.0601521, 7.7302866, 10.0681171, -2.3292780, 2.3298655
9: -6.3480220, -2.8028464, -6.3509588, -2.7878101, -3.2882094, 3.2682176

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4598

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3998366, upper bound: 1.3953314
time: 9.99 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3998366, upper bound: 1.3998366
time: 9.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 34.71 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 34.71
Output dim: 8, lower bound: -1.3976653, upper bound: 1.3976648
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 34.71
Output dim: 8, lower bound: -1.3976653, upper bound: 1.3998449
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 34.71
Output dim: 8, lower bound: -1.3998366, upper bound: 1.3953314
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 34.71
Output dim: 8, lower bound: -1.3998366, upper bound: 1.3998366

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -6.1727219, -1.9947731, -6.1727219, -1.9947731, -3.6883249, 3.6883254
1: -12.2193813, -8.9918165, -12.2193813, -8.9918165, -3.2181458, 3.2181463
2: -5.6021371, -2.2392807, -5.6021371, -2.2392807, -3.0790062, 3.0790071
3: -5.3690162, -1.5078959, -5.3690162, -1.5078959, -3.8611202, 3.8611202
4: -11.5189686, -7.5783043, -11.5189686, -7.5783043, -3.7487001, 3.7487001
5: -6.2878017, -3.0851064, -6.2878017, -3.0851064, -2.9530692, 2.9530697
6: -12.4260111, -8.6976948, -12.4260111, -8.6976948, -3.3509560, 3.3509560
7: -8.1650391, -4.6791401, -8.1650391, -4.6791401, -3.4858990, 3.4858990
8: 7.7414150, 10.0551043, 7.7414150, 10.0551043, -2.3136892, 2.3136892
9: -6.3445268, -2.8050718, -6.3445268, -2.8050718, -3.2572255, 3.2572260

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3952411, upper bound: 1.3974343
time: 5.45 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976621, upper bound: 1.3976621
time: 4.64 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -6.2037158, -1.9477437, -6.1727219, -1.9947731, -3.7186422, 3.7355227
1: -12.2469511, -8.9744186, -12.2193813, -8.9918165, -3.2465587, 3.2363787
2: -5.6229401, -2.1984048, -5.6021371, -2.2392807, -3.1007338, 3.1166103
3: -5.3863273, -1.5006928, -5.3690162, -1.5078959, -3.8784313, 3.8683233
4: -11.5466595, -7.5620070, -11.5189686, -7.5783043, -3.7758493, 3.7653041
5: -6.3025022, -3.0794778, -6.2878017, -3.0851064, -2.9676814, 2.9591761
6: -12.4390497, -8.6935787, -12.4260111, -8.6976948, -3.3627033, 3.3566232
7: -8.1877203, -4.6719475, -8.1650391, -4.6791401, -3.5085802, 3.4930916
8: 7.7302866, 10.0681171, 7.7414150, 10.0551043, -2.3248177, 2.3267021
9: -6.3509588, -2.7878101, -6.3445268, -2.8050718, -3.2633052, 3.2743578

Time for backsubstitution: 14.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3952412, upper bound: 1.3996288
time: 5.27 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976621, upper bound: 1.3998423
time: 5.20 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -6.1713676, -2.0445552, -6.2023287, -1.9513302, -3.7044239, 3.6668568
1: -12.2015362, -8.9949579, -12.2446041, -8.9745970, -3.2200260, 3.2368016
2: -5.5637584, -2.2480173, -5.6189113, -2.1990452, -3.0779080, 3.0805912
3: -5.3126221, -1.5175335, -5.3822131, -1.5014780, -3.8111441, 3.8646796
4: -11.5084848, -7.6976829, -11.5459213, -7.5712228, -3.7149134, 3.6559319
5: -6.2785149, -3.1275840, -6.3018746, -3.0827184, -2.9625773, 2.9291506
6: -12.4118366, -8.7015848, -12.4379044, -8.6939220, -3.3661661, 3.3639035
7: -8.1549702, -4.6843405, -8.1867514, -4.6728735, -3.4820967, 3.5024109
8: 7.7500987, 10.0356722, 7.7309561, 10.0663948, -2.3162961, 2.3047161
9: -6.3348751, -2.8942502, -6.3502808, -2.7942817, -3.2537117, 3.1764822

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976566, upper bound: 1.3931537
time: 7.07 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976566, upper bound: 1.3953317
time: 9.86 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -6.1948929, -1.9940336, -6.2037158, -1.9477437, -3.7571783, 3.6990147
1: -12.2345247, -8.9912519, -12.2469511, -8.9744186, -3.2601061, 3.2459936
2: -5.6206713, -2.2354507, -5.6229401, -2.1984048, -3.0987554, 3.1025519
3: -5.3708873, -1.5042095, -5.3863273, -1.5006928, -3.8701944, 3.8821177
4: -11.5237999, -7.5678253, -11.5466595, -7.5620070, -3.7695503, 3.6988811
5: -6.2900934, -3.0817852, -6.3025022, -3.0794778, -2.9786100, 2.9623418
6: -12.4278774, -8.6957130, -12.4390497, -8.6935787, -3.3784428, 3.3713541
7: -8.1703548, -4.6722736, -8.1877203, -4.6719475, -3.4984074, 3.5154467
8: 7.7388401, 10.0601511, 7.7302866, 10.0681171, -2.3292770, 2.3298645
9: -6.3480225, -2.8028493, -6.3509588, -2.7878101, -3.2882104, 3.2206688

Time for backsubstitution: 14.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976566, upper bound: 1.3976561
time: 5.61 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976566, upper bound: 1.3998362
time: 5.49 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.29 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 8, lower bound: -1.3952411, upper bound: 1.3974343
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 8, lower bound: -1.3976621, upper bound: 1.3976621
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 8, lower bound: -1.3952412, upper bound: 1.3996288
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 8, lower bound: -1.3976621, upper bound: 1.3998423
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 8, lower bound: -1.3976566, upper bound: 1.3931537
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 8, lower bound: -1.3976566, upper bound: 1.3953317
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 8, lower bound: -1.3976566, upper bound: 1.3976561
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 8, lower bound: -1.3976566, upper bound: 1.3998362

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -6.1621284, -2.0187166, -6.1718965, -2.0003936, -3.6700754, 3.6633391
1: -12.2056417, -8.9981718, -12.2166500, -8.9930582, -3.2026491, 3.2079697
2: -5.5960608, -2.2605476, -5.6015115, -2.2440352, -3.0667725, 3.0565205
3: -5.3506393, -1.5171015, -5.3650966, -1.5089812, -3.8416581, 3.8479950
4: -11.5021973, -7.5870790, -11.5159073, -7.5793452, -3.7304678, 3.7352657
5: -6.2791467, -3.1065421, -6.2870655, -3.0901256, -2.9378395, 2.9295807
6: -12.4182100, -8.7106514, -12.4249249, -8.7006226, -3.3402691, 3.3362927
7: -8.1455431, -4.6824942, -8.1612616, -4.6797004, -3.4658427, 3.4787674
8: 7.7531829, 10.0477810, 7.7439928, 10.0539417, -2.3007588, 2.3037882
9: -6.3328576, -2.8177021, -6.3431859, -2.8075867, -3.2431049, 3.2424955

Time for backsubstitution: 14.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4598

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3907373, upper bound: 1.3974254
time: 6.42 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3952323, upper bound: 1.3974259
time: 4.92 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -6.1727219, -1.9947796, -6.1727219, -1.9947731, -3.6883249, 3.6748381
1: -12.2193813, -8.9918156, -12.2193813, -8.9918165, -3.2181430, 3.2219224
2: -5.6021366, -2.2392843, -5.6021371, -2.2392807, -3.0790062, 3.0768943
3: -5.3690143, -1.5078967, -5.3690162, -1.5078959, -3.8611183, 3.8611195
4: -11.5189676, -7.5783076, -11.5189686, -7.5783043, -3.7468052, 3.7486992
5: -6.2878041, -3.0851116, -6.2878017, -3.0851064, -2.9530683, 2.9407248
6: -12.4260111, -8.6976995, -12.4260111, -8.6976948, -3.3513732, 3.3509531
7: -8.1650362, -4.6791410, -8.1650391, -4.6791401, -3.4858961, 3.4858980
8: 7.7414179, 10.0551033, 7.7414150, 10.0551043, -2.3136864, 2.3136883
9: -6.3445263, -2.8050733, -6.3445268, -2.8050718, -3.2572246, 3.2539721

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4598

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3931508, upper bound: 1.3976531
time: 7.17 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976533, upper bound: 1.3976535
time: 5.14 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -6.1931095, -1.9716358, -6.1718965, -2.0003936, -3.7003736, 3.7105732
1: -12.2332678, -8.9807749, -12.2166500, -8.9930582, -3.2310877, 3.2261825
2: -5.6168265, -2.2197113, -5.6015115, -2.2440352, -3.0884342, 3.0941176
3: -5.3681040, -1.5099022, -5.3650966, -1.5089812, -3.8591228, 3.8551943
4: -11.5300026, -7.5708055, -11.5159073, -7.5793452, -3.7577705, 3.7518826
5: -6.2938461, -3.1008787, -6.2870655, -3.0901256, -2.9524059, 2.9357438
6: -12.4312420, -8.7065058, -12.4249249, -8.7006226, -3.3519564, 3.3420434
7: -8.1683264, -4.6752987, -8.1612616, -4.6797004, -3.4886260, 3.4859629
8: 7.7420254, 10.0607967, 7.7439928, 10.0539417, -2.3119164, 2.3168039
9: -6.3392596, -2.8001897, -6.3431859, -2.8075867, -3.2491541, 3.2598934

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4598

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3907373, upper bound: 1.3996206
time: 7.01 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3952324, upper bound: 1.3996204
time: 4.58 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -6.2037163, -1.9477482, -6.1727219, -1.9947731, -3.7186413, 3.7221732
1: -12.2469482, -8.9744177, -12.2193813, -8.9918165, -3.2465558, 3.2401562
2: -5.6229410, -2.1984084, -5.6021371, -2.2392807, -3.1007328, 3.1144981
3: -5.3863225, -1.5006955, -5.3690162, -1.5078959, -3.8784266, 3.8683207
4: -11.5466585, -7.5620060, -11.5189686, -7.5783043, -3.7739525, 3.7653031
5: -6.3025031, -3.0794840, -6.2878017, -3.0851064, -2.9676795, 2.9468303
6: -12.4390488, -8.6935835, -12.4260111, -8.6976948, -3.3631215, 3.3566203
7: -8.1877155, -4.6719475, -8.1650391, -4.6791401, -3.5085754, 3.4930916
8: 7.7302909, 10.0681171, 7.7414150, 10.0551043, -2.3248134, 2.3267021
9: -6.3509579, -2.7878122, -6.3445268, -2.8050718, -3.2633033, 3.2711020

Time for backsubstitution: 14.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 6182

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4598

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3931508, upper bound: 1.3998338
time: 5.57 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976533, upper bound: 1.3998335
time: 4.69 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -6.1491609, -2.0453005, -6.2023287, -1.9513302, -3.6823559, 3.6672487
1: -12.1862774, -8.9955158, -12.2446041, -8.9745970, -3.2047472, 3.2363400
2: -5.5452142, -2.2517624, -5.6189113, -2.1990452, -3.0590863, 3.0769391
3: -5.3107929, -1.5212238, -5.3822131, -1.5014780, -3.8093150, 3.8609893
4: -11.5037012, -7.7081842, -11.5459213, -7.5712228, -3.7098656, 3.6454730
5: -6.2762842, -3.1309223, -6.3018746, -3.0827184, -2.9431224, 2.9208527
6: -12.4100361, -8.7035484, -12.4379044, -8.6939220, -3.3418384, 3.3553295
7: -8.1497459, -4.6912050, -8.1867514, -4.6728735, -3.4768724, 3.4955463
8: 7.7526388, 10.0305758, 7.7309561, 10.0663948, -2.3137560, 2.2996197
9: -6.3313637, -2.8964851, -6.3502808, -2.7942817, -3.2452059, 3.1715655

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of IS_B2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3974279, upper bound: 1.3907373
time: 6.25 seconds

## Relational analysis of IS_B2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976535, upper bound: 1.3931507
time: 6.53 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -6.1803427, -1.9982710, -6.2023287, -1.9513302, -3.7114325, 3.6886778
1: -12.2138081, -8.9781189, -12.2446041, -8.9745970, -3.2293339, 3.2508497
2: -5.5660176, -2.2110264, -5.6189113, -2.1990452, -3.0538826, 3.0873518
3: -5.3278856, -1.5139890, -5.3822131, -1.5014780, -3.8264077, 3.8682241
4: -11.5313330, -7.6918812, -11.5459213, -7.5712228, -3.7196321, 3.6559811
5: -6.2908683, -3.1252961, -6.3018746, -3.0827184, -2.9703403, 2.9395204
6: -12.4227295, -8.6994686, -12.4379044, -8.6939220, -3.3714085, 3.3789434
7: -8.1723261, -4.6840158, -8.1867514, -4.6728735, -3.4994526, 3.5027356
8: 7.7415352, 10.0435953, 7.7309561, 10.0663948, -2.3248596, 2.3126392
9: -6.3378611, -2.8792691, -6.3502808, -2.7942817, -3.2562895, 3.1989799

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3963870, upper bound: 1.3953259
time: 6.08 seconds

## Relational analysis of IS_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976489, upper bound: 1.3953234
time: 9.54 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -6.1727200, -1.9947746, -6.2037158, -1.9477437, -3.7351313, 3.6994081
1: -12.2193804, -8.9918165, -12.2469511, -8.9744186, -3.2449617, 3.2455254
2: -5.6021338, -2.2392812, -5.6229401, -2.1984048, -3.0799327, 3.1007333
3: -5.3690138, -1.5078962, -5.3863273, -1.5006928, -3.8683209, 3.8784311
4: -11.5189686, -7.5783119, -11.5466595, -7.5620070, -3.7653046, 3.6884298
5: -6.2878008, -3.0851073, -6.3025022, -3.0794778, -2.9591751, 2.9540467
6: -12.4260092, -8.6976938, -12.4390497, -8.6935787, -3.3541236, 3.3627038
7: -8.1650400, -4.6791410, -8.1877203, -4.6719475, -3.4930925, 3.5085793
8: 7.7414160, 10.0551033, 7.7302866, 10.0681171, -2.3267012, 2.3248167
9: -6.3445263, -2.8050785, -6.3509588, -2.7878101, -3.2743573, 3.2157559

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3974256, upper bound: 1.3952322
time: 8.77 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976535, upper bound: 1.3976528
time: 5.56 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -6.2037172, -1.9477446, -6.2037158, -1.9477437, -3.7400770, 3.7208433
1: -12.2469473, -8.9744167, -12.2469511, -8.9744186, -3.2725286, 3.2600427
2: -5.6229367, -2.1984067, -5.6229401, -2.1984048, -3.0748715, 3.1113124
3: -5.3863230, -1.5006945, -5.3863273, -1.5006928, -3.8856301, 3.8856328
4: -11.5466595, -7.5620122, -11.5466595, -7.5620070, -3.7863741, 3.6989551
5: -6.3025017, -3.0794821, -6.3025022, -3.0794778, -2.9863644, 2.9727321
6: -12.4390469, -8.6935787, -12.4390497, -8.6935787, -3.3836670, 3.3863511
7: -8.1877213, -4.6719503, -8.1877203, -4.6719475, -3.5157738, 3.5157700
8: 7.7302885, 10.0681162, 7.7302866, 10.0681171, -2.3378286, 2.3378296
9: -6.3509603, -2.7878156, -6.3509588, -2.7878101, -3.2907863, 3.2432365

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_B2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3963869, upper bound: 1.3998302
time: 7.74 seconds

## Relational analysis of IS_B2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976489, upper bound: 1.3998286
time: 4.62 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.96 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 8, lower bound: -1.3907373, upper bound: 1.3974254
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 8, lower bound: -1.3952323, upper bound: 1.3974259
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 8, lower bound: -1.3931508, upper bound: 1.3976531
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 8, lower bound: -1.3976533, upper bound: 1.3976535
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 8, lower bound: -1.3907373, upper bound: 1.3996206
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 8, lower bound: -1.3952324, upper bound: 1.3996204
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 8, lower bound: -1.3931508, upper bound: 1.3998338
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 8, lower bound: -1.3976533, upper bound: 1.3998335
IS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 8, lower bound: -1.3974279, upper bound: 1.3907373
IS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 8, lower bound: -1.3976535, upper bound: 1.3931507
IS_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 8, lower bound: -1.3963870, upper bound: 1.3953259
IS_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 8, lower bound: -1.3976489, upper bound: 1.3953234
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 8, lower bound: -1.3974256, upper bound: 1.3952322
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 8, lower bound: -1.3976535, upper bound: 1.3976528
IS_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 8, lower bound: -1.3963869, upper bound: 1.3998302
IS_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 8, lower bound: -1.3976489, upper bound: 1.3998286

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -6.1607356, -2.0222998, -6.1483459, -2.0509090, -3.6186848, 3.6372013
1: -12.2032928, -8.9983482, -12.1835365, -8.9967461, -3.1924229, 3.1763473
2: -5.5920324, -2.2611718, -5.5445910, -2.2565093, -3.0404720, 2.9991083
3: -5.3465424, -1.5178857, -5.3068733, -1.5223014, -3.8242409, 3.7889876
4: -11.5014572, -7.5962906, -11.5006332, -7.7092185, -3.6001053, 3.6791577
5: -6.2785263, -3.1097846, -6.2755432, -3.1359482, -2.8910179, 2.9135175
6: -12.4170885, -8.7109909, -12.4089537, -8.7064791, -3.3329086, 3.3215132
7: -8.1445866, -4.6834202, -8.1459913, -4.6917667, -3.4528198, 3.4625711
8: 7.7538476, 10.0460577, 7.7552128, 10.0294161, -2.2755685, 2.2908449
9: -6.3321867, -2.8241708, -6.3300352, -2.8990014, -3.1513605, 3.2176719

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of IS_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3907395, upper bound: 1.3952321
time: 7.97 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3907373, upper bound: 1.3974257
time: 7.43 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -6.1621284, -2.0187166, -6.1718965, -2.0003972, -3.6508408, 3.6633377
1: -12.2056417, -8.9981718, -12.2166500, -8.9930563, -3.2016182, 3.2184782
2: -5.5960608, -2.2605476, -5.6015067, -2.2440348, -3.0667715, 3.0200725
3: -5.3506393, -1.5171015, -5.3650923, -1.5089810, -3.8416584, 3.8479908
4: -11.5021973, -7.5870790, -11.5159082, -7.5793486, -3.6430626, 3.7352662
5: -6.2791467, -3.1065421, -6.2870646, -3.0901270, -2.9242039, 2.9295807
6: -12.4182100, -8.7106514, -12.4249239, -8.7006226, -3.3402686, 3.3338017
7: -8.1455431, -4.6824942, -8.1612606, -4.6797028, -3.4658403, 3.4787664
8: 7.7531829, 10.0477810, 7.7439938, 10.0539408, -2.3007579, 2.3037872
9: -6.3328576, -2.8177021, -6.3431873, -2.8075919, -3.1955547, 3.2424936

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of IS_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3952324, upper bound: 1.3952324
time: 4.60 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3952324, upper bound: 1.3974260
time: 4.59 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -6.1713228, -1.9983652, -6.1491609, -2.0453005, -3.6369252, 3.6486788
1: -12.2170343, -8.9919958, -12.1862774, -8.9955158, -3.2079110, 3.1902905
2: -5.5981083, -2.2399116, -5.5452142, -2.2517624, -3.0543385, 3.0194774
3: -5.3649158, -1.5086789, -5.3107929, -1.5212238, -3.8436921, 3.8021140
4: -11.5182314, -7.5875225, -11.5037012, -7.7081842, -3.6164417, 3.6923661
5: -6.2871828, -3.0883508, -6.2762842, -3.1309223, -2.9062586, 2.9246707
6: -12.4248924, -8.6980400, -12.4100361, -8.7035484, -3.3440170, 3.3361683
7: -8.1640730, -4.6800718, -8.1497459, -4.6912050, -3.4728680, 3.4696741
8: 7.7420845, 10.0533772, 7.7526388, 10.0305758, -2.2884912, 2.3007383
9: -6.3438511, -2.8115432, -6.3313637, -2.8964851, -3.1654806, 3.2291956

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3908529, upper bound: 1.3976501
time: 5.74 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3931476, upper bound: 1.3976502
time: 9.46 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -6.1727219, -1.9947796, -6.1727200, -1.9947746, -3.6690893, 3.6748371
1: -12.2193813, -8.9918156, -12.2193804, -8.9918165, -3.2171097, 3.2275648
2: -5.6021366, -2.2392843, -5.6021338, -2.2392812, -3.0790062, 3.0404582
3: -5.3690143, -1.5078967, -5.3690138, -1.5078962, -3.8611181, 3.8611171
4: -11.5189676, -7.5783076, -11.5189686, -7.5783119, -3.6594000, 3.7486997
5: -6.2878041, -3.0851116, -6.2878008, -3.0851073, -2.9394350, 2.9407244
6: -12.4260111, -8.6976995, -12.4260092, -8.6976938, -3.3513722, 3.3484530
7: -8.1650362, -4.6791410, -8.1650400, -4.6791410, -3.4858952, 3.4858990
8: 7.7414179, 10.0551033, 7.7414160, 10.0551033, -2.3136854, 2.3136873
9: -6.3445263, -2.8050733, -6.3445263, -2.8050785, -3.2096744, 3.2539697

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3953404, upper bound: 1.3976500
time: 7.75 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976501, upper bound: 1.3976503
time: 4.64 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -6.1917272, -1.9752176, -6.1483459, -2.0509090, -3.6489916, 3.6574521
1: -12.2309160, -8.9809532, -12.1835365, -8.9967461, -3.2208796, 3.1945591
2: -5.6127987, -2.2203436, -5.5445910, -2.2565093, -3.0602751, 3.0365953
3: -5.3639927, -1.5106828, -5.3068733, -1.5223014, -3.8416913, 3.7961905
4: -11.5292587, -7.5800180, -11.5006332, -7.7092185, -3.6273937, 3.6947360
5: -6.2932177, -3.1041207, -6.2755432, -3.1359482, -2.9055681, 2.9196806
6: -12.4300985, -8.7068491, -12.4089537, -8.7064791, -3.3445778, 3.3272634
7: -8.1673603, -4.6762276, -8.1459913, -4.6917667, -3.4755936, 3.4697638
8: 7.7426920, 10.0590734, 7.7552128, 10.0294161, -2.2867241, 2.3038607
9: -6.3385925, -2.8066611, -6.3300352, -2.8990014, -3.1574144, 3.2307296

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of IS_B1_A2_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3907373, upper bound: 1.3974325
time: 9.64 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2

### Relational analysis result of IS_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3907395, upper bound: 1.3996202
time: 7.41 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -6.1931095, -1.9716358, -6.1718965, -2.0003972, -3.6811390, 3.7101769
1: -12.2332678, -8.9807749, -12.2166500, -8.9930563, -3.2300568, 3.2358751
2: -5.6168265, -2.2197113, -5.6015067, -2.2440348, -3.0884333, 3.0574288
3: -5.3681040, -1.5099022, -5.3650923, -1.5089810, -3.8591230, 3.8551900
4: -11.5300026, -7.5708055, -11.5159082, -7.5793486, -3.6703520, 3.7518830
5: -6.2938461, -3.1008787, -6.2870646, -3.0901270, -2.9387732, 2.9357433
6: -12.4312420, -8.7065058, -12.4249239, -8.7006226, -3.3519559, 3.3395510
7: -8.1683264, -4.6752987, -8.1612606, -4.6797028, -3.4886236, 3.4859619
8: 7.7420254, 10.0607967, 7.7439938, 10.0539408, -2.3119154, 2.3168030
9: -6.3392596, -2.8001897, -6.3431873, -2.8075919, -3.2016039, 3.2598915

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of IS_B1_A2_A1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3952324, upper bound: 1.3974328
time: 5.02 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3952324, upper bound: 1.3996203
time: 4.67 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -6.2023306, -1.9513338, -6.1491609, -2.0453005, -3.6672492, 3.6681840
1: -12.2446012, -8.9745989, -12.1862774, -8.9955158, -3.2363372, 3.2085228
2: -5.6189108, -2.1990471, -5.5452142, -2.2517624, -3.0741844, 3.0569739
3: -5.3822103, -1.5014770, -5.3107929, -1.5212238, -3.8609865, 3.8093159
4: -11.5459185, -7.5712223, -11.5037012, -7.7081842, -3.6435776, 3.7078986
5: -6.3018746, -3.0827236, -6.2762842, -3.1309223, -2.9208517, 2.9307775
6: -12.4379044, -8.6939240, -12.4100361, -8.7035484, -3.3557463, 3.3418345
7: -8.1867476, -4.6728735, -8.1497459, -4.6912050, -3.4955425, 3.4768724
8: 7.7309585, 10.0663929, 7.7526388, 10.0305758, -2.2996173, 2.3137541
9: -6.3502808, -2.7942815, -6.3313637, -2.8964851, -3.1715651, 3.2419443

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3908529, upper bound: 1.3998302
time: 8.61 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3931476, upper bound: 1.3998314
time: 10.54 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -6.2037163, -1.9477482, -6.1727200, -1.9947746, -3.6994085, 3.7209539
1: -12.2469482, -8.9744177, -12.2193804, -8.9918165, -3.2455225, 3.2449627
2: -5.6229410, -2.1984084, -5.6021338, -2.2392812, -3.1007318, 3.0778208
3: -5.3863225, -1.5006955, -5.3690138, -1.5078962, -3.8784263, 3.8683183
4: -11.5466585, -7.5620060, -11.5189686, -7.5783119, -3.6865349, 3.7653036
5: -6.3025031, -3.0794840, -6.2878008, -3.0851073, -2.9540462, 2.9468298
6: -12.4390488, -8.6935835, -12.4260092, -8.6976938, -3.3631206, 3.3541198
7: -8.1877155, -4.6719475, -8.1650400, -4.6791410, -3.5085745, 3.4930925
8: 7.7302909, 10.0681171, 7.7414160, 10.0551033, -2.3248124, 2.3267012
9: -6.3509579, -2.7878122, -6.3445263, -2.8050785, -3.2157531, 3.2711000

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3953404, upper bound: 1.3998309
time: 99.50 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976501, upper bound: 1.3998304
time: 4.67 seconds

## BFS IS instance: IS_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -6.1483459, -2.0509090, -6.1917272, -1.9752176, -3.6574521, 3.6489921
1: -12.1835365, -8.9967461, -12.2309160, -8.9809532, -3.1945591, 3.2208800
2: -5.5445910, -2.2565093, -5.6127987, -2.2203436, -3.0365953, 3.0602756
3: -5.3068733, -1.5223014, -5.3639927, -1.5106828, -3.7961905, 3.8416913
4: -11.5006332, -7.7092185, -11.5292587, -7.5800180, -3.6947365, 3.6273937
5: -6.2755432, -3.1359482, -6.2932177, -3.1041207, -2.9196806, 2.9055686
6: -12.4089537, -8.7064791, -12.4300985, -8.7068491, -3.3272634, 3.3445783
7: -8.1459913, -4.6917667, -8.1673603, -4.6762276, -3.4697638, 3.4755936
8: 7.7552128, 10.0294161, 7.7426920, 10.0590734, -2.3038607, 2.2867241
9: -6.3300352, -2.8990014, -6.3385925, -2.8066611, -3.2307301, 3.1574144

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_B2_A1_A1_B1_A1

### Relational analysis result of IS_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3974325, upper bound: 1.3907367
time: 6.80 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2

### Relational analysis result of IS_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3974325, upper bound: 1.3907362
time: 8.97 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 30.35 seconds
IS_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 8, lower bound: -1.3907395, upper bound: 1.3952321
IS_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 8, lower bound: -1.3907373, upper bound: 1.3974257
IS_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 8, lower bound: -1.3952324, upper bound: 1.3952324
IS_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 8, lower bound: -1.3952324, upper bound: 1.3974260
IS_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 8, lower bound: -1.3908529, upper bound: 1.3976501
IS_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 8, lower bound: -1.3931476, upper bound: 1.3976502
IS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 8, lower bound: -1.3953404, upper bound: 1.3976500
IS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 8, lower bound: -1.3976501, upper bound: 1.3976503
IS_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 8, lower bound: -1.3907373, upper bound: 1.3974325
IS_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 8, lower bound: -1.3907395, upper bound: 1.3996202
IS_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 8, lower bound: -1.3952324, upper bound: 1.3974328
IS_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 8, lower bound: -1.3952324, upper bound: 1.3996203
IS_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 8, lower bound: -1.3908529, upper bound: 1.3998302
IS_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 8, lower bound: -1.3931476, upper bound: 1.3998314
IS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 8, lower bound: -1.3953404, upper bound: 1.3998309
IS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 8, lower bound: -1.3976501, upper bound: 1.3998304
IS_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 8, lower bound: -1.3974325, upper bound: 1.3907367
IS_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 8, lower bound: -1.3974325, upper bound: 1.3907362
IS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.35
Output dim: 8, lower bound: -1.3976535, upper bound: 1.3931507
IS_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 30.35
Output dim: 8, lower bound: -1.3963870, upper bound: 1.3953259
IS_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 30.35
Output dim: 8, lower bound: -1.3976489, upper bound: 1.3953234
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.35
Output dim: 8, lower bound: -1.3974256, upper bound: 1.3952322
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.35
Output dim: 8, lower bound: -1.3976535, upper bound: 1.3976528
IS_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 30.35
Output dim: 8, lower bound: -1.3963869, upper bound: 1.3998302
IS_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 30.35
Output dim: 8, lower bound: -1.3976489, upper bound: 1.3998286
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.321315288543701
rel_dist={8: [-1.3998489967280836, 1.3998484636799233]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354390, upper bound: 1.1366276
time: 5.16 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1366286, upper bound: 1.1366276
time: 6.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.54 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 11.54
Output dim: 8, lower bound: -1.1354390, upper bound: 1.1366276
IS_B2, status: Status.UNKNOWN, split count: 1, time: 11.54
Output dim: 8, lower bound: -1.1366286, upper bound: 1.1366276

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -6.1861491, -1.9943221, -6.1727219, -1.9947731, -3.3429108, 3.3301506
1: -12.2285404, -8.9914751, -12.2193813, -8.9918165, -2.9247584, 2.9159784
2: -5.6133595, -2.2369547, -5.6021371, -2.2392807, -2.8686924, 2.8593802
3: -5.3701553, -1.5056684, -5.3690162, -1.5078959, -3.6380453, 3.6391187
4: -11.5219069, -7.5719557, -11.5189686, -7.5783043, -3.4289656, 3.4325461
5: -6.2891932, -3.0831032, -6.2878017, -3.0851064, -2.7388673, 2.7401748
6: -12.4271469, -8.6964922, -12.4260111, -8.6976948, -3.0321398, 3.0328193
7: -8.1682634, -4.6749802, -8.1650391, -4.6791401, -3.4891233, 3.4900589
8: 7.7398520, 10.0581551, 7.7414150, 10.0551043, -2.2770505, 2.2787924
9: -6.3466406, -2.8037207, -6.3445268, -2.8050718, -3.0112491, 3.0105162

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354390, upper bound: 1.1354380
time: 5.20 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354413, upper bound: 1.1366278
time: 5.37 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -6.1948862, -1.9940336, -6.2037158, -1.9477437, -3.3949785, 3.3549504
1: -12.2345200, -8.9912529, -12.2469511, -8.9744186, -2.9489927, 2.9437985
2: -5.6206689, -2.2354527, -5.6229401, -2.1984048, -2.9099236, 2.8754559
3: -5.3708882, -1.5042102, -5.3863273, -1.5006928, -3.6543493, 3.6573377
4: -11.5237999, -7.5678225, -11.5466595, -7.5620070, -3.4459934, 3.4638171
5: -6.2900934, -3.0817842, -6.3025022, -3.0794778, -2.7583208, 2.7580466
6: -12.4278812, -8.6957159, -12.4390497, -8.6935787, -3.0553398, 3.0480003
7: -8.1703539, -4.6722755, -8.1877203, -4.6719475, -3.4984064, 3.5154448
8: 7.7388411, 10.0601492, 7.7302866, 10.0681171, -2.2934885, 2.3039932
9: -6.3480215, -2.8028460, -6.3509588, -2.7878101, -3.0375161, 3.0185270

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1349249, upper bound: 1.1362031
time: 7.16 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1366268, upper bound: 1.1366262
time: 5.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 27.61 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 27.61
Output dim: 8, lower bound: -1.1354390, upper bound: 1.1354380
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 27.61
Output dim: 8, lower bound: -1.1354413, upper bound: 1.1366278
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 27.61
Output dim: 8, lower bound: -1.1349249, upper bound: 1.1362031
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 27.61
Output dim: 8, lower bound: -1.1366268, upper bound: 1.1366262

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -6.1727219, -1.9947731, -6.1727219, -1.9947731, -3.3295612, 3.3295617
1: -12.2193813, -8.9918165, -12.2193813, -8.9918165, -2.9155746, 2.9155741
2: -5.6021371, -2.2392807, -5.6021371, -2.2392807, -2.8573055, 2.8573046
3: -5.3690162, -1.5078959, -5.3690162, -1.5078959, -3.6346550, 3.6346555
4: -11.5189686, -7.5783043, -11.5189686, -7.5783043, -3.4262180, 3.4262180
5: -6.2878017, -3.0851064, -6.2878017, -3.0851064, -2.7351432, 2.7351432
6: -12.4260111, -8.6976948, -12.4260111, -8.6976948, -3.0276041, 3.0276041
7: -8.1650391, -4.6791401, -8.1650391, -4.6791401, -3.4858990, 3.4858990
8: 7.7414150, 10.0551043, 7.7414150, 10.0551043, -2.2733483, 2.2733483
9: -6.3445268, -2.8050718, -6.3445268, -2.8050718, -3.0075374, 3.0075374

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337325, upper bound: 1.1350070
time: 5.92 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354372, upper bound: 1.1354363
time: 7.61 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -6.2037158, -1.9477437, -6.1727219, -1.9947731, -3.3598795, 3.3729324
1: -12.2469511, -8.9744186, -12.2193813, -8.9918165, -2.9439874, 2.9338074
2: -5.6229401, -2.1984048, -5.6021371, -2.2392807, -2.8790312, 2.8911057
3: -5.3863273, -1.5006928, -5.3690162, -1.5078959, -3.6499805, 3.6430268
4: -11.5466595, -7.5620070, -11.5189686, -7.5783043, -3.4533672, 3.4428220
5: -6.3025022, -3.0794778, -6.2878017, -3.0851064, -2.7497535, 2.7412496
6: -12.4390497, -8.6935787, -12.4260111, -8.6976948, -3.0393524, 3.0332713
7: -8.1877203, -4.6719475, -8.1650391, -4.6791401, -3.5085802, 3.4930916
8: 7.7302866, 10.0681171, 7.7414150, 10.0551043, -2.2850327, 2.2873855
9: -6.3509588, -2.7878101, -6.3445268, -2.8050718, -3.0136161, 3.0246696

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337302, upper bound: 1.1362004
time: 5.47 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354372, upper bound: 1.1366261
time: 7.27 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -6.1842823, -2.0179698, -6.2023263, -1.9571679, -3.3675938, 3.3292723
1: -12.2208128, -8.9975929, -12.2424030, -8.9765024, -2.9328508, 2.9319515
2: -5.6145787, -2.2567024, -5.6218777, -2.2064123, -2.8902650, 2.8524480
3: -5.3525052, -1.5134180, -5.3798018, -1.5025318, -3.6333857, 3.6407981
4: -11.5071239, -7.5765953, -11.5415573, -7.5637732, -3.4270058, 3.4483008
5: -6.2814331, -3.1031871, -6.3012605, -3.0878873, -2.7396259, 2.7339859
6: -12.4200764, -8.7086506, -12.4372120, -8.6984825, -3.0425339, 3.0327053
7: -8.1508942, -4.6756277, -8.1814299, -4.6728897, -3.4780045, 3.5058022
8: 7.7506061, 10.0528069, 7.7346039, 10.0661707, -2.2789569, 2.2894449
9: -6.3363323, -2.8154986, -6.3486814, -2.7919412, -3.0217080, 3.0027771

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4598

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1348984, upper bound: 1.1320058
time: 6.03 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1349156, upper bound: 1.1361908
time: 15.05 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -6.1948853, -1.9940379, -6.2037172, -1.9477448, -3.3899603, 3.3383746
1: -12.2345181, -8.9912548, -12.2469511, -8.9744167, -2.9489460, 2.9471493
2: -5.6206694, -2.2354538, -5.6229410, -2.1984053, -2.9071875, 2.8729725
3: -5.3708868, -1.5042100, -5.3863249, -1.5006940, -3.6468992, 3.6573372
4: -11.5237961, -7.5678225, -11.5466614, -7.5620060, -3.4437637, 3.4638157
5: -6.2900915, -3.0817890, -6.3025022, -3.0794792, -2.7583208, 2.7435226
6: -12.4278774, -8.6957178, -12.4390507, -8.6935797, -3.0557079, 3.0479908
7: -8.1703520, -4.6722741, -8.1877203, -4.6719470, -3.4984050, 3.5154462
8: 7.7388439, 10.0601482, 7.7302876, 10.0681181, -2.2934394, 2.3073092
9: -6.3480196, -2.8028491, -6.3509588, -2.7878110, -3.0375166, 3.0146971

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 540

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4598

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1366061, upper bound: 1.1324305
time: 14.73 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1366198, upper bound: 1.1366169
time: 5.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 35.37 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 35.37
Output dim: 8, lower bound: -1.1337325, upper bound: 1.1350070
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 35.37
Output dim: 8, lower bound: -1.1354372, upper bound: 1.1354363
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 35.37
Output dim: 8, lower bound: -1.1337302, upper bound: 1.1362004
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 35.37
Output dim: 8, lower bound: -1.1354372, upper bound: 1.1366261
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 35.37
Output dim: 8, lower bound: -1.1348984, upper bound: 1.1320058
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 35.37
Output dim: 8, lower bound: -1.1349156, upper bound: 1.1361908
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 35.37
Output dim: 8, lower bound: -1.1366061, upper bound: 1.1324305
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 35.37
Output dim: 8, lower bound: -1.1366198, upper bound: 1.1366169

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -6.1621284, -2.0187166, -6.1713324, -2.0042167, -3.3070126, 3.3038907
1: -12.2056417, -8.9981718, -12.2148075, -8.9938974, -2.8994694, 2.9036474
2: -5.5960608, -2.2605476, -5.6010852, -2.2472689, -2.8417826, 2.8343096
3: -5.3506393, -1.5171015, -5.3624363, -1.5097334, -3.6136465, 3.6180701
4: -11.5021973, -7.5870790, -11.5138397, -7.5800576, -3.4071007, 3.4106550
5: -6.2791467, -3.1065421, -6.2865629, -3.0935383, -2.7164192, 2.7110505
6: -12.4182100, -8.7106514, -12.4241772, -8.7026138, -3.0147934, 3.0123315
7: -8.1455431, -4.6824942, -8.1587095, -4.6800838, -3.4654593, 3.4762154
8: 7.7531829, 10.0477810, 7.7457385, 10.0531464, -2.2588310, 2.2587900
9: -6.3328576, -2.8177021, -6.3422656, -2.8092957, -2.9916544, 2.9918237

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4598

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1295260, upper bound: 1.1349763
time: 9.14 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337231, upper bound: 1.1349973
time: 6.39 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -6.1727219, -1.9947796, -6.1727214, -1.9947739, -3.3295603, 3.3129430
1: -12.2193813, -8.9918156, -12.2193813, -8.9918156, -2.9155254, 2.9189253
2: -5.6021366, -2.2392843, -5.6021376, -2.2392812, -2.8573036, 2.8548198
3: -5.3690143, -1.5078967, -5.3690162, -1.5078957, -3.6272030, 3.6346536
4: -11.5189676, -7.5783076, -11.5189686, -7.5783057, -3.4239902, 3.4262166
5: -6.2878041, -3.0851116, -6.2878017, -3.0851068, -2.7351427, 2.7206192
6: -12.4260111, -8.6976995, -12.4260101, -8.6976948, -3.0279746, 3.0275965
7: -8.1650362, -4.6791410, -8.1650400, -4.6791401, -3.4858961, 3.4858990
8: 7.7414179, 10.0551033, 7.7414150, 10.0551033, -2.2733026, 2.2766652
9: -6.3445263, -2.8050733, -6.3445263, -2.8050735, -3.0075369, 3.0037079

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 540

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4598

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1312299, upper bound: 1.1354138
time: 6.86 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354278, upper bound: 1.1354272
time: 8.15 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -6.1931095, -1.9716358, -6.1713324, -2.0042167, -3.3373117, 3.3473005
1: -12.2332678, -8.9807749, -12.2148075, -8.9938974, -2.9279070, 2.9218602
2: -5.6168265, -2.2197113, -5.6010852, -2.2472689, -2.8634443, 2.8681097
3: -5.3681040, -1.5099022, -5.3624363, -1.5097334, -3.6291351, 3.6264386
4: -11.5300026, -7.5708055, -11.5138397, -7.5800576, -3.4344025, 3.4272718
5: -6.2938461, -3.1008787, -6.2865629, -3.0935383, -2.7309856, 2.7172132
6: -12.4312420, -8.7065058, -12.4241772, -8.7026138, -3.0264807, 3.0180821
7: -8.1683264, -4.6752987, -8.1587095, -4.6800838, -3.4882426, 3.4834108
8: 7.7420254, 10.0607967, 7.7457385, 10.0531464, -2.2705798, 2.2727704
9: -6.3392596, -2.8001897, -6.3422656, -2.8092957, -2.9977036, 3.0092216

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4598

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1295260, upper bound: 1.1361778
time: 7.09 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337208, upper bound: 1.1361909
time: 5.38 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -6.2037163, -1.9477482, -6.1727214, -1.9947739, -3.3598776, 3.3555756
1: -12.2469482, -8.9744177, -12.2193813, -8.9918156, -2.9439383, 2.9371591
2: -5.6229410, -2.1984084, -5.6021376, -2.2392812, -2.8790302, 2.8884318
3: -5.3863225, -1.5006955, -5.3690162, -1.5078957, -3.6425285, 3.6430249
4: -11.5466585, -7.5620060, -11.5189686, -7.5783057, -3.4511375, 3.4428210
5: -6.3025031, -3.0794840, -6.2878017, -3.0851068, -2.7497530, 2.7267246
6: -12.4390488, -8.6935835, -12.4260101, -8.6976948, -3.0397229, 3.0332637
7: -8.1877155, -4.6719475, -8.1650400, -4.6791401, -3.5085754, 3.4930925
8: 7.7302909, 10.0681171, 7.7414150, 10.0551033, -2.2849855, 2.2907016
9: -6.3509579, -2.7878122, -6.3445263, -2.8050735, -3.0136156, 3.0208378

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 540

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4598

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1312322, upper bound: 1.1366085
time: 7.14 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354301, upper bound: 1.1366167
time: 8.46 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -6.1607952, -2.0684364, -6.1996474, -1.9640787, -3.3069324, 3.2767258
1: -12.1877956, -9.0012569, -12.2378778, -8.9768486, -2.9010935, 2.9173689
2: -5.5576701, -2.2692349, -5.6141133, -2.2076426, -2.8321691, 2.8195825
3: -5.2942634, -1.5267146, -5.3718786, -1.5040445, -3.5736961, 3.6190910
4: -11.4917793, -7.7064548, -11.5401230, -7.5815334, -3.3487692, 3.3172903
5: -6.2698278, -3.1490240, -6.3000474, -3.0941334, -2.7102933, 2.6864548
6: -12.4040470, -8.7145405, -12.4350100, -8.6991453, -3.0274601, 3.0238118
7: -8.1355877, -4.6876955, -8.1795616, -4.6746621, -3.4609256, 3.4918661
8: 7.7618303, 10.0283537, 7.7358942, 10.0628462, -2.2617760, 2.2664342
9: -6.3232341, -2.9069145, -6.3473883, -2.8044136, -2.9693041, 2.9104319

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_B2_A1_A1_A1

### Relational analysis result of IS_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1336973, upper bound: 1.1308020
time: 6.16 seconds

## Relational analysis of IS_B2_A1_A1_A2

### Relational analysis result of IS_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1336974, upper bound: 1.1320033
time: 8.22 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -6.1842818, -2.0179727, -6.2023263, -1.9571679, -3.3657618, 3.3066416
1: -12.2208128, -8.9975929, -12.2424030, -8.9765024, -2.9534545, 2.9309182
2: -5.6145754, -2.2567015, -5.6218777, -2.2064123, -2.8460269, 2.8524475
3: -5.3525033, -1.5134194, -5.3798018, -1.5025318, -3.6139297, 3.6407986
4: -11.5071220, -7.5766029, -11.5415573, -7.5637732, -3.4270048, 3.3454490
5: -6.2814322, -3.1031904, -6.3012605, -3.0878873, -2.7396240, 2.7179441
6: -12.4200735, -8.7086525, -12.4372120, -8.6984825, -3.0382252, 3.0327039
7: -8.1508942, -4.6756291, -8.1814299, -4.6728897, -3.4780045, 3.5058007
8: 7.7506061, 10.0528049, 7.7346039, 10.0661707, -2.2787118, 2.2903359
9: -6.3363290, -2.8155019, -6.3486814, -2.7919412, -3.0217061, 2.9468632

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_B2_A1_A2_B1

### Relational analysis result of IS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1349154, upper bound: 1.1344591
time: 5.48 seconds

## Relational analysis of IS_B2_A1_A2_B2

### Relational analysis result of IS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1349132, upper bound: 1.1361891
time: 11.43 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -6.1713567, -2.0445609, -6.2010345, -1.9546561, -3.3292017, 3.2858119
1: -12.2015295, -8.9949579, -12.2424288, -8.9747658, -2.9171891, 2.9325442
2: -5.5637498, -2.2480223, -5.6151748, -2.1996398, -2.8490896, 2.8399487
3: -5.3126192, -1.5175374, -5.3784027, -1.5022063, -3.5871687, 3.6362305
4: -11.5084820, -7.6976886, -11.5452299, -7.5797691, -3.3652682, 3.3328080
5: -6.2785158, -3.1275902, -6.3012905, -3.0857253, -2.7307172, 2.6960306
6: -12.4118385, -8.7015877, -12.4368477, -8.6942425, -3.0406189, 3.0390682
7: -8.1549644, -4.6843419, -8.1858482, -4.6737204, -3.4812441, 3.5015063
8: 7.7501011, 10.0356693, 7.7315807, 10.0647945, -2.2762370, 2.2842712
9: -6.3348703, -2.8942518, -6.3496509, -2.8002841, -2.9859738, 2.9223580

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_B2_A2_A1_A1

### Relational analysis result of IS_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354142, upper bound: 1.1312298
time: 8.08 seconds

## Relational analysis of IS_B2_A2_A1_A2

### Relational analysis result of IS_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354164, upper bound: 1.1324312
time: 8.95 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -6.1948843, -1.9940395, -6.2037172, -1.9477448, -3.3881273, 3.3157430
1: -12.2345161, -8.9912539, -12.2469511, -8.9744167, -2.9695287, 2.9461174
2: -5.6206641, -2.2354555, -5.6229410, -2.1984053, -2.8629818, 2.8729715
3: -5.3708820, -1.5042102, -5.3863249, -1.5006940, -3.6274462, 3.6573367
4: -11.5237980, -7.5678277, -11.5466614, -7.5620060, -3.4437637, 3.3609638
5: -6.2900929, -3.0817924, -6.3025022, -3.0794792, -2.7583208, 2.7274790
6: -12.4278784, -8.6957178, -12.4390507, -8.6935797, -3.0513749, 3.0479922
7: -8.1703520, -4.6722746, -8.1877203, -4.6719470, -3.4984050, 3.5154457
8: 7.7388458, 10.0601482, 7.7302876, 10.0681181, -2.2931924, 2.3082395
9: -6.3480201, -2.8028531, -6.3509588, -2.7878110, -3.0375147, 2.9587803

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1366152, upper bound: 1.1348594
time: 5.80 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1366151, upper bound: 1.1366144
time: 5.71 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.41 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.41
Output dim: 8, lower bound: -1.1295260, upper bound: 1.1349763
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.41
Output dim: 8, lower bound: -1.1337231, upper bound: 1.1349973
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.41
Output dim: 8, lower bound: -1.1312299, upper bound: 1.1354138
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.41
Output dim: 8, lower bound: -1.1354278, upper bound: 1.1354272
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.41
Output dim: 8, lower bound: -1.1295260, upper bound: 1.1361778
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.41
Output dim: 8, lower bound: -1.1337208, upper bound: 1.1361909
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.41
Output dim: 8, lower bound: -1.1312322, upper bound: 1.1366085
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.41
Output dim: 8, lower bound: -1.1354301, upper bound: 1.1366167
IS_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 26.41
Output dim: 8, lower bound: -1.1336973, upper bound: 1.1308020
IS_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 26.41
Output dim: 8, lower bound: -1.1336974, upper bound: 1.1320033
IS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.41
Output dim: 8, lower bound: -1.1349154, upper bound: 1.1344591
IS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.41
Output dim: 8, lower bound: -1.1349132, upper bound: 1.1361891
IS_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 26.41
Output dim: 8, lower bound: -1.1354142, upper bound: 1.1312298
IS_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 26.41
Output dim: 8, lower bound: -1.1354164, upper bound: 1.1324312
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.41
Output dim: 8, lower bound: -1.1366152, upper bound: 1.1348594
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.41
Output dim: 8, lower bound: -1.1366151, upper bound: 1.1366144

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -6.1594329, -2.0256190, -6.1477838, -2.0547254, -3.2544022, 3.2744789
1: -12.2011185, -8.9985142, -12.1816854, -8.9975824, -2.8848448, 2.8717661
2: -5.5882969, -2.2617567, -5.5441647, -2.2597377, -2.8048291, 2.7762833
3: -5.3427482, -1.5186157, -5.3042145, -1.5230460, -3.5946026, 3.5583310
4: -11.5007696, -7.6048379, -11.4985590, -7.7099309, -3.2761087, 3.3309388
5: -6.2779479, -3.1127901, -6.2750349, -3.1393685, -2.6689367, 2.6919589
6: -12.4160538, -8.7113123, -12.4082108, -8.7084732, -3.0059748, 2.9972420
7: -8.1436958, -4.6842642, -8.1434526, -4.6921468, -3.4515491, 3.4591885
8: 7.7544680, 10.0444622, 7.7569532, 10.0286245, -2.2357478, 2.2415962
9: -6.3315601, -2.8301716, -6.3291240, -2.9007156, -2.8993044, 2.9517655

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_B1_A1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1278458, upper bound: 1.1349739
time: 6.81 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1295213, upper bound: 1.1349737
time: 9.11 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -6.1621284, -2.0187166, -6.1713305, -2.0042195, -3.2843819, 3.3038912
1: -12.2056417, -8.9981718, -12.2148056, -8.9938974, -2.8984385, 2.9242134
2: -5.5960608, -2.2605476, -5.6010804, -2.2472692, -2.8417826, 2.7903700
3: -5.3506393, -1.5171015, -5.3624334, -1.5097342, -3.6136446, 3.5986137
4: -11.5021973, -7.5870790, -11.5138397, -7.5800629, -3.3042612, 3.4106541
5: -6.2791467, -3.1065421, -6.2865610, -3.0935407, -2.7003746, 2.7110510
6: -12.4182100, -8.7106514, -12.4241762, -8.7026119, -3.0147943, 3.0080113
7: -8.1455431, -4.6824942, -8.1587095, -4.6800857, -3.4654574, 3.4762154
8: 7.7531829, 10.0477810, 7.7457376, 10.0531464, -2.2597194, 2.2585411
9: -6.3328576, -2.8177021, -6.3422651, -2.8093007, -2.9357376, 2.9918246

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_B1_A1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1319850, upper bound: 1.1349979
time: 8.21 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337184, upper bound: 1.1349950
time: 5.55 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -6.1700172, -2.0016904, -6.1491594, -2.0453019, -3.2769337, 3.2834954
1: -12.2148609, -8.9921646, -12.1862774, -8.9955158, -2.9008980, 2.8870826
2: -5.5943718, -2.2404966, -5.5452151, -2.2517624, -2.8217850, 2.7967882
3: -5.3611202, -1.5094144, -5.3107934, -1.5212231, -3.6081409, 3.5749049
4: -11.5175438, -7.5960698, -11.5036993, -7.7081842, -3.2930012, 3.3461590
5: -6.2866049, -3.0913544, -6.2762852, -3.1309214, -2.6876812, 2.7015419
6: -12.4238539, -8.6983557, -12.4100351, -8.7035494, -3.0191603, 3.0124989
7: -8.1631784, -4.6809130, -8.1497440, -4.6912060, -3.4719725, 3.4688311
8: 7.7427058, 10.0517788, 7.7526402, 10.0305767, -2.2502041, 2.2594597
9: -6.3432217, -2.8175416, -6.3313646, -2.8964853, -2.9151840, 2.9636226

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1295720, upper bound: 1.1354141
time: 6.94 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1312275, upper bound: 1.1354120
time: 6.75 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -6.1727219, -1.9947796, -6.1727204, -1.9947724, -3.3069282, 3.3129416
1: -12.2193813, -8.9918156, -12.2193823, -8.9918175, -2.9144945, 2.9394355
2: -5.6021366, -2.2392843, -5.6021338, -2.2392831, -2.8573036, 2.8108983
3: -5.3690143, -1.5078967, -5.3690128, -1.5078952, -3.6272030, 3.6151981
4: -11.5189676, -7.5783076, -11.5189676, -7.5783119, -3.3211508, 3.4262171
5: -6.2878041, -3.0851116, -6.2877998, -3.0851083, -2.7190990, 2.7206187
6: -12.4260111, -8.6976995, -12.4260111, -8.6976938, -3.0279727, 3.0232615
7: -8.1650362, -4.6791410, -8.1650391, -4.6791410, -3.4858952, 3.4858980
8: 7.7414179, 10.0551033, 7.7414155, 10.0551023, -2.2742524, 2.2764168
9: -6.3445263, -2.8050733, -6.3445263, -2.8050773, -2.9516206, 3.0037060

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1336710, upper bound: 1.1354248
time: 5.16 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354254, upper bound: 1.1354246
time: 10.56 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -6.1904364, -1.9785430, -6.1477838, -2.0547254, -3.2847166, 3.2865796
1: -12.2287407, -8.9811192, -12.1816854, -8.9975824, -2.9133186, 2.8899784
2: -5.6090636, -2.2209358, -5.5441647, -2.2597377, -2.8182650, 2.8099909
3: -5.3601832, -1.5114110, -5.3042145, -1.5230460, -3.6100473, 3.5667024
4: -11.5285673, -7.5885653, -11.4985590, -7.7099309, -3.3033838, 3.3429565
5: -6.2926311, -3.1071281, -6.2750349, -3.1393685, -2.6834717, 2.6981220
6: -12.4290428, -8.7071705, -12.4082108, -8.7084732, -3.0176296, 3.0029950
7: -8.1664648, -4.6770720, -8.1434526, -4.6921468, -3.4743180, 3.4663806
8: 7.7433152, 10.0574760, 7.7569532, 10.0286245, -2.2474933, 2.2556005
9: -6.3379722, -2.8126643, -6.3291240, -2.9007156, -2.9053602, 2.9648232

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1278458, upper bound: 1.1361754
time: 5.93 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1295214, upper bound: 1.1361736
time: 5.92 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -6.1931095, -1.9716358, -6.1713305, -2.0042195, -3.3146791, 3.3454685
1: -12.2332678, -8.9807749, -12.2148056, -8.9938974, -2.9268751, 2.9424114
2: -5.6168265, -2.2197113, -5.6010804, -2.2472692, -2.8634443, 2.8238831
3: -5.3681040, -1.5099022, -5.3624334, -1.5097342, -3.6291332, 3.6069822
4: -11.5300026, -7.5708055, -11.5138397, -7.5800629, -3.3315516, 3.4272709
5: -6.2938461, -3.1008787, -6.2865610, -3.0935407, -2.7149439, 2.7172136
6: -12.4312420, -8.7065058, -12.4241762, -8.7026119, -3.0264807, 3.0137610
7: -8.1683264, -4.6752987, -8.1587095, -4.6800857, -3.4882407, 3.4834108
8: 7.7420254, 10.0607967, 7.7457376, 10.0531464, -2.2714343, 2.2725220
9: -6.3392596, -2.8001897, -6.3422651, -2.8093007, -2.9417877, 3.0092230

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_B1_A2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1319873, upper bound: 1.1361913
time: 10.27 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337184, upper bound: 1.1361885
time: 5.46 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -6.2010326, -1.9546580, -6.1491594, -2.0453019, -3.3072653, 3.2947917
1: -12.2424278, -8.9747639, -12.1862774, -8.9955158, -2.9293385, 2.9053149
2: -5.6151748, -2.1996424, -5.5452151, -2.2517624, -2.8352594, 2.8303230
3: -5.3784008, -1.5022094, -5.3107934, -1.5212231, -3.6234226, 3.5832810
4: -11.5452290, -7.5797706, -11.5036993, -7.7081842, -3.3201227, 3.3581390
5: -6.3012896, -3.0857301, -6.2762852, -3.1309214, -2.7022581, 2.7076492
6: -12.4368458, -8.6942453, -12.4100351, -8.7035494, -3.0308733, 3.0181656
7: -8.1858463, -4.6737204, -8.1497440, -4.6912060, -3.4946404, 3.4760237
8: 7.7315817, 10.0647955, 7.7526402, 10.0305767, -2.2618828, 2.2735214
9: -6.3496509, -2.8002858, -6.3313646, -2.8964853, -2.9212723, 2.9763694

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 540

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1295697, upper bound: 1.1366039
time: 8.05 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1312274, upper bound: 1.1366042
time: 7.60 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -6.2037163, -1.9477482, -6.1727204, -1.9947724, -3.3372464, 3.3537433
1: -12.2469482, -8.9744177, -12.2193823, -8.9918175, -2.9429073, 2.9576550
2: -5.6229410, -2.1984084, -5.6021338, -2.2392831, -2.8790302, 2.8442218
3: -5.3863225, -1.5006955, -5.3690128, -1.5078952, -3.6425266, 3.6235704
4: -11.5466585, -7.5620060, -11.5189676, -7.5783119, -3.3482866, 3.4428215
5: -6.3025031, -3.0794840, -6.2877998, -3.0851083, -2.7337093, 2.7267241
6: -12.4390488, -8.6935835, -12.4260111, -8.6976938, -3.0397220, 3.0289278
7: -8.1877155, -4.6719475, -8.1650391, -4.6791410, -3.5085745, 3.4930916
8: 7.7302909, 10.0681171, 7.7414155, 10.0551023, -2.2859001, 2.2904534
9: -6.3509579, -2.7878122, -6.3445263, -2.8050773, -2.9576993, 3.0208359

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1336733, upper bound: 1.1366145
time: 5.81 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354254, upper bound: 1.1366139
time: 7.37 seconds

## BFS IS instance: IS_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -6.1386061, -2.0691857, -6.1996474, -1.9640787, -3.2848740, 3.2816534
1: -12.1725044, -9.0018311, -12.2378778, -8.9768486, -2.8858023, 2.9175930
2: -5.5391531, -2.2729948, -5.6141133, -2.2076426, -2.8133631, 2.8149652
3: -5.2924409, -1.5303969, -5.3718786, -1.5040445, -3.5623121, 3.6117573
4: -11.4868984, -7.7169476, -11.5401230, -7.5815334, -3.3411484, 3.3068442
5: -6.2676044, -3.1523933, -6.3000474, -3.0941334, -2.7017155, 2.6781120
6: -12.4022579, -8.7165222, -12.4350100, -8.6991453, -3.0054250, 3.0151849
7: -8.1303310, -4.6945581, -8.1795616, -4.6746621, -3.4556689, 3.4850035
8: 7.7643685, 10.0232782, 7.7358942, 10.0628462, -2.2557149, 2.2474432
9: -6.3197479, -2.9091289, -6.3473883, -2.8044136, -2.9618282, 2.9055433

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_B2_A1_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1336972, upper bound: 1.1291250
time: 8.50 seconds

## Relational analysis of IS_B2_A1_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1336951, upper bound: 1.1308012
time: 9.62 seconds

## BFS IS instance: IS_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -6.1697745, -2.0221109, -6.1996474, -1.9640787, -3.3093853, 3.2985907
1: -12.2000933, -8.9844389, -12.2378778, -8.9768486, -2.9097981, 2.9314260
2: -5.5599146, -2.2323010, -5.6141133, -2.2076426, -2.8105803, 2.8263786
3: -5.3096838, -1.5231681, -5.3718786, -1.5040445, -3.5836658, 3.6236753
4: -11.5146437, -7.7006731, -11.5401230, -7.5815334, -3.3534660, 3.3162942
5: -6.2821846, -3.1467295, -6.3000474, -3.0941334, -2.7181315, 2.6945295
6: -12.4149418, -8.7124119, -12.4350100, -8.6991453, -3.0326958, 3.0367088
7: -8.1530094, -4.6873684, -8.1795616, -4.6746621, -3.4783473, 3.4921932
8: 7.7532396, 10.0362997, 7.7358942, 10.0628462, -2.2812991, 2.2752776
9: -6.3262243, -2.8916621, -6.3473883, -2.8044136, -2.9718742, 2.9322214

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_B2_A1_A1_A2_B1

### Relational analysis result of IS_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1336950, upper bound: 1.1303307
time: 9.97 seconds

## Relational analysis of IS_B2_A1_A1_A2_B2

### Relational analysis result of IS_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1336972, upper bound: 1.1320015
time: 8.15 seconds

## BFS IS instance: IS_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -6.1768165, -2.0278521, -6.1623201, -1.9814258, -3.3428602, 3.2322788
1: -12.2101336, -8.9989204, -12.2123442, -8.9933872, -2.9441605, 2.8954058
2: -5.6129942, -2.2636118, -5.5962439, -2.2301848, -2.8545380, 2.8256345
3: -5.3503857, -1.5239525, -5.3520288, -1.5303955, -3.5740919, 3.6101151
4: -11.5015383, -7.5852938, -11.5052185, -7.5861101, -3.4044380, 3.1384306
5: -6.2535539, -3.1054106, -6.2330117, -3.1334147, -2.6481743, 2.6399274
6: -12.4071016, -8.7112007, -12.4031916, -8.7314262, -2.9598775, 2.9880338
7: -8.1432323, -4.6770315, -8.1547508, -4.6870775, -3.4561548, 3.4777193
8: 7.7531714, 10.0490274, 7.7465248, 10.0453377, -2.2507606, 2.2505350
9: -6.3300662, -2.8302228, -6.3060875, -2.8253868, -2.9743834, 2.8837934

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of IS_B2_A1_A2_B1_B1

### Relational analysis result of IS_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1349131, upper bound: 1.1331796
time: 5.56 seconds

## Relational analysis of IS_B2_A1_A2_B1_B2

### Relational analysis result of IS_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1349131, upper bound: 1.1344584
time: 5.47 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.62 seconds
IS_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1278458, upper bound: 1.1349739
IS_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1295213, upper bound: 1.1349737
IS_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1319850, upper bound: 1.1349979
IS_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1337184, upper bound: 1.1349950
IS_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1295720, upper bound: 1.1354141
IS_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1312275, upper bound: 1.1354120
IS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1336710, upper bound: 1.1354248
IS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1354254, upper bound: 1.1354246
IS_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1278458, upper bound: 1.1361754
IS_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1295214, upper bound: 1.1361736
IS_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1319873, upper bound: 1.1361913
IS_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1337184, upper bound: 1.1361885
IS_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1295697, upper bound: 1.1366039
IS_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1312274, upper bound: 1.1366042
IS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1336733, upper bound: 1.1366145
IS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1354254, upper bound: 1.1366139
IS_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1336972, upper bound: 1.1291250
IS_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1336951, upper bound: 1.1308012
IS_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1336950, upper bound: 1.1303307
IS_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1336972, upper bound: 1.1320015
IS_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1349131, upper bound: 1.1331796
IS_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 25.62
Output dim: 8, lower bound: -1.1349131, upper bound: 1.1344584
IS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.62
Output dim: 8, lower bound: -1.1349132, upper bound: 1.1361891
IS_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 25.62
Output dim: 8, lower bound: -1.1354142, upper bound: 1.1312298
IS_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 25.62
Output dim: 8, lower bound: -1.1354164, upper bound: 1.1324312
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.62
Output dim: 8, lower bound: -1.1366152, upper bound: 1.1348594
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.62
Output dim: 8, lower bound: -1.1366151, upper bound: 1.1366144
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.2884626388549805
rel_dist={8: [-1.136630455303056, 1.136628973869513]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405787, upper bound: 1.0396755
time: 5.61 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405812, upper bound: 1.0405779
time: 5.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.69 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.69
Output dim: 8, lower bound: -1.0405787, upper bound: 1.0396755
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.69
Output dim: 8, lower bound: -1.0405812, upper bound: 1.0405779

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -6.1727219, -1.9947731, -6.1833220, -1.9944172, -3.2104406, 3.2205076
1: -12.2193813, -8.9918165, -12.2266035, -8.9915438, -2.8150358, 2.8219624
2: -5.6021371, -2.2392807, -5.6109924, -2.2374434, -2.7850437, 2.7923899
3: -5.3690162, -1.5078959, -5.3699155, -1.5061395, -3.5507679, 3.5499167
4: -11.5189686, -7.5783043, -11.5212889, -7.5732942, -3.3237181, 3.3208961
5: -6.2878017, -3.0851064, -6.2889009, -3.0835266, -2.6664743, 2.6654391
6: -12.4260111, -8.6976948, -12.4269085, -8.6967449, -2.9239378, 2.9233999
7: -8.1650391, -4.6791401, -8.1675873, -4.6758556, -3.4891834, 3.4884472
8: 7.7414150, 10.0551043, 7.7401800, 10.0575113, -2.2146707, 2.2132983
9: -6.3445268, -2.8050718, -6.3461947, -2.8040037, -2.9266605, 2.9272366

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400870, upper bound: 1.0382999
time: 9.45 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405773, upper bound: 1.0396737
time: 5.89 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.2037158, -1.9477437, -6.1948805, -1.9940329, -3.2338514, 3.2736344
1: -12.2469511, -8.9744186, -12.2345171, -8.9912529, -2.8427243, 2.8481340
2: -5.6229401, -2.1984048, -5.6206656, -2.2354522, -2.7997561, 2.8347549
3: -5.3863273, -1.5006928, -5.3708878, -1.5042093, -3.5699272, 3.5663614
4: -11.5466595, -7.5620070, -11.5238008, -7.5678220, -3.3563213, 3.3381410
5: -6.3025022, -3.0794778, -6.2900920, -3.0817828, -2.6854019, 2.6848931
6: -12.4390497, -8.6935787, -12.4278793, -8.6957150, -2.9402161, 2.9468050
7: -8.1877203, -4.6719475, -8.1703529, -4.6722746, -3.5154457, 3.4984055
8: 7.7302866, 10.0681171, 7.7388411, 10.0601492, -2.2405248, 2.2305138
9: -6.3509588, -2.7878101, -6.3480196, -2.8028471, -2.9352970, 2.9539523

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400870, upper bound: 1.0391547
time: 5.73 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405798, upper bound: 1.0405761
time: 5.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.02 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.02
Output dim: 8, lower bound: -1.0400870, upper bound: 1.0382999
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.02
Output dim: 8, lower bound: -1.0405773, upper bound: 1.0396737
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.02
Output dim: 8, lower bound: -1.0400870, upper bound: 1.0391547
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.02
Output dim: 8, lower bound: -1.0405798, upper bound: 1.0405761

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -6.1710367, -2.0061800, -6.1727219, -2.0183597, -3.1844082, 3.1957393
1: -12.2138615, -8.9943295, -12.2128782, -8.9978943, -2.8022733, 2.8055277
2: -5.6008663, -2.2489285, -5.6049080, -2.2587013, -2.7617912, 2.7751746
3: -5.3610725, -1.5101206, -5.3515344, -1.5153463, -3.5328560, 3.5284429
4: -11.5127735, -7.5804257, -11.5045643, -7.5820694, -3.3070612, 3.3013711
5: -6.2863035, -3.0952902, -6.2802439, -3.1049490, -2.6420932, 2.6449256
6: -12.4237890, -8.7036324, -12.4191065, -8.7096920, -2.9083767, 2.9094825
7: -8.1574068, -4.6802788, -8.1481094, -4.6792102, -3.4781966, 3.4678307
8: 7.7466345, 10.0527382, 7.7519455, 10.0501804, -2.1989174, 2.1984017
9: -6.3417902, -2.8101749, -6.3345146, -2.8166432, -2.9104252, 2.9104376

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392551, upper bound: 1.0383031
time: 6.87 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392527, upper bound: 1.0383031
time: 6.85 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -6.1727204, -1.9947748, -6.1833167, -1.9944220, -3.1927967, 3.2205038
1: -12.2193851, -8.9918156, -12.2266026, -8.9915457, -2.8182430, 2.8217888
2: -5.6021376, -2.2392817, -5.6109934, -2.2374463, -2.7824345, 2.7923884
3: -5.3690157, -1.5078959, -5.3699121, -1.5061395, -3.5507660, 3.5420938
4: -11.5189695, -7.5783062, -11.5212898, -7.5732951, -3.3237181, 3.3185577
5: -6.2878017, -3.0851068, -6.2888994, -3.0835323, -2.6512237, 2.6654387
6: -12.4260101, -8.6976967, -12.4269085, -8.6967468, -2.9239149, 2.9237523
7: -8.1650381, -4.6791401, -8.1675825, -4.6758556, -3.4891825, 3.4884424
8: 7.7414160, 10.0551033, 7.7401838, 10.0575104, -2.2178464, 2.2131274
9: -6.3445272, -2.8050723, -6.3461928, -2.8040063, -2.9226360, 2.9272361

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 540

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396869, upper bound: 1.0396744
time: 6.08 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396869, upper bound: 1.0396775
time: 7.92 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.2020330, -1.9591279, -6.1842804, -2.0179718, -3.2078180, 3.2449820
1: -12.2414646, -8.9769354, -12.2208099, -8.9975910, -2.8300409, 2.8316722
2: -5.6216569, -2.2080774, -5.6145754, -2.2567012, -2.7764826, 2.8138132
3: -5.3784490, -1.5029228, -5.3525047, -1.5134175, -3.5520716, 3.5449138
4: -11.5405006, -7.5641451, -11.5071220, -7.5765972, -3.3397293, 3.3186936
5: -6.3009987, -3.0896358, -6.2814331, -3.1031885, -2.6610260, 2.6644068
6: -12.4368258, -8.6995010, -12.4200745, -8.7086506, -2.9246001, 2.9329157
7: -8.1801319, -4.6730866, -8.1508942, -4.6756268, -3.5045052, 3.4778075
8: 7.7354970, 10.0657616, 7.7506075, 10.0528059, -2.2247949, 2.2155981
9: -6.3482046, -2.7927980, -6.3363318, -2.8154981, -2.9190378, 2.9372568

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4598

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0366397, upper bound: 1.0388455
time: 5.95 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400774, upper bound: 1.0391477
time: 7.24 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.2037172, -1.9477458, -6.1948800, -1.9940369, -3.2162299, 3.2686152
1: -12.2469501, -8.9744158, -12.2345142, -8.9912548, -2.8459339, 2.8479614
2: -5.6229410, -2.1984062, -5.6206651, -2.2354560, -2.7971478, 2.8320191
3: -5.3863239, -1.5006950, -5.3708844, -1.5042109, -3.5699244, 3.5585399
4: -11.5466614, -7.5620060, -11.5237980, -7.5678225, -3.3563204, 3.3358016
5: -6.3025012, -3.0794802, -6.2900929, -3.0817895, -2.6701527, 2.6848912
6: -12.4390497, -8.6935787, -12.4278774, -8.6957169, -2.9401922, 2.9471598
7: -8.1877193, -4.6719465, -8.1703510, -4.6722755, -3.5154438, 3.4984045
8: 7.7302885, 10.0681171, 7.7388439, 10.0601482, -2.2436996, 2.2303419
9: -6.3509579, -2.7878113, -6.3480206, -2.8028488, -2.9312730, 2.9539518

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 540

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4598

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0371259, upper bound: 1.0402622
time: 6.65 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405700, upper bound: 1.0405689
time: 5.21 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.65 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 26.65
Output dim: 8, lower bound: -1.0392551, upper bound: 1.0383031
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 26.65
Output dim: 8, lower bound: -1.0392527, upper bound: 1.0383031
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 26.65
Output dim: 8, lower bound: -1.0396869, upper bound: 1.0396744
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 26.65
Output dim: 8, lower bound: -1.0396869, upper bound: 1.0396775
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 26.65
Output dim: 8, lower bound: -1.0366397, upper bound: 1.0388455
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 26.65
Output dim: 8, lower bound: -1.0400774, upper bound: 1.0391477
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 26.65
Output dim: 8, lower bound: -1.0371259, upper bound: 1.0402622
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 26.65
Output dim: 8, lower bound: -1.0405700, upper bound: 1.0405689

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -6.1710367, -2.0061800, -6.1621284, -2.0187166, -3.1839495, 3.1852098
1: -12.2138615, -8.9943295, -12.2056417, -8.9981718, -2.8019691, 2.7982941
2: -5.6008663, -2.2489285, -5.5960608, -2.2605476, -2.7601490, 2.7661943
3: -5.3610725, -1.5101206, -5.3506393, -1.5171015, -3.5293331, 3.5257521
4: -11.5127735, -7.5804257, -11.5021973, -7.5870790, -3.3020716, 3.2991476
5: -6.2863035, -3.0952902, -6.2791467, -3.1065421, -2.6380978, 2.6419845
6: -12.4237890, -8.7036324, -12.4182100, -8.7106514, -2.9042325, 2.9059196
7: -8.1574068, -4.6802788, -8.1455431, -4.6824942, -3.4749126, 3.4652643
8: 7.7466345, 10.0527382, 7.7531829, 10.0477810, -2.1946311, 2.1954784
9: -6.3417902, -2.8101749, -6.3328576, -2.8177021, -2.9080877, 2.9075203

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4598

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0389547, upper bound: 1.0348681
time: 5.86 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392456, upper bound: 1.0382958
time: 5.92 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -6.1710367, -2.0061800, -6.1931095, -1.9716358, -3.2256050, 3.2155089
1: -12.2138615, -8.9943295, -12.2332678, -8.9807749, -2.8201818, 2.8267317
2: -5.6008663, -2.2489285, -5.6168265, -2.2197113, -2.7926831, 2.7878556
3: -5.3610725, -1.5101206, -5.3681040, -1.5099022, -3.5377026, 3.5412407
4: -11.5127735, -7.5804257, -11.5300026, -7.5708055, -3.3186893, 3.3264503
5: -6.2863035, -3.0952902, -6.2938461, -3.1008787, -2.6442614, 2.6565518
6: -12.4237890, -8.7036324, -12.4312420, -8.7065058, -2.9099832, 2.9176064
7: -8.1574068, -4.6802788, -8.1683264, -4.6752987, -3.4821081, 3.4880476
8: 7.7466345, 10.0527382, 7.7420254, 10.0607967, -2.2086120, 2.2072272
9: -6.3417902, -2.8101749, -6.3392596, -2.8001897, -2.9254856, 2.9135695

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4598

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0389547, upper bound: 1.0348681
time: 6.67 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392480, upper bound: 1.0382958
time: 6.32 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -6.1727204, -1.9947748, -6.1727219, -1.9947796, -3.1923094, 3.2099714
1: -12.2193851, -8.9918156, -12.2193813, -8.9918156, -2.8179259, 2.8145428
2: -5.6021376, -2.2392817, -5.6021366, -2.2392843, -2.7807951, 2.7834029
3: -5.3690157, -1.5078959, -5.3690143, -1.5078967, -3.5472422, 3.5394211
4: -11.5189695, -7.5783062, -11.5189676, -7.5783076, -3.3187237, 3.3163853
5: -6.2878017, -3.0851068, -6.2878041, -3.0851116, -2.6472507, 2.6624990
6: -12.4260101, -8.6976967, -12.4260111, -8.6976995, -2.9197998, 2.9201722
7: -8.1650381, -4.6791401, -8.1650362, -4.6791410, -3.4858971, 3.4858961
8: 7.7414160, 10.0551033, 7.7414179, 10.0551033, -2.2135515, 2.2102056
9: -6.3445272, -2.8050723, -6.3445263, -2.8050733, -2.9202833, 2.9243069

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 540

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4598

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0393655, upper bound: 1.0362236
time: 7.16 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396775, upper bound: 1.0396672
time: 6.09 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -6.1727204, -1.9947748, -6.2037163, -1.9477482, -3.2330885, 3.2402883
1: -12.2193851, -8.9918156, -12.2469482, -8.9744177, -2.8361588, 2.8429556
2: -5.6021376, -2.2392817, -5.6229410, -2.1984084, -2.8126397, 2.8051291
3: -5.3690157, -1.5078959, -5.3863225, -1.5006955, -3.5556135, 3.5547462
4: -11.5189695, -7.5783062, -11.5466585, -7.5620060, -3.3353281, 3.3435330
5: -6.2878017, -3.0851068, -6.3025031, -3.0794840, -2.6533575, 2.6771102
6: -12.4260101, -8.6976967, -12.4390488, -8.6935835, -2.9254656, 2.9319210
7: -8.1650381, -4.6791401, -8.1877155, -4.6719475, -3.4930906, 3.5085754
8: 7.7414160, 10.0551033, 7.7302909, 10.0681171, -2.2275882, 2.2218885
9: -6.3445272, -2.8050723, -6.3509579, -2.7878122, -2.9374132, 2.9303856

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 540

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4598

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0393655, upper bound: 1.0362215
time: 6.79 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396775, upper bound: 1.0396676
time: 6.74 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -6.1981812, -1.9683313, -6.1607909, -2.0684371, -3.1539850, 3.1821456
1: -12.2353153, -8.9773941, -12.1877928, -9.0012560, -2.8122830, 2.7997565
2: -5.6113291, -2.2098494, -5.5576668, -2.2692361, -2.7398930, 2.7551804
3: -5.3678675, -1.5050743, -5.2942648, -1.5267148, -3.5259581, 3.4847951
4: -11.5385647, -7.5877733, -11.4917784, -7.7064562, -3.2082214, 3.2323170
5: -6.2992916, -3.0979595, -6.2698293, -3.1490231, -2.6128430, 2.6326075
6: -12.4336758, -8.7003889, -12.4040470, -8.7145424, -2.9144793, 2.9176211
7: -8.1776333, -4.6758013, -8.1355867, -4.6876941, -3.4899392, 3.4597855
8: 7.7374606, 10.0613365, 7.7618313, 10.0283527, -2.2011213, 2.1965628
9: -6.3464556, -2.8093944, -6.3232327, -2.9069152, -2.8262577, 2.8799419

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0353390, upper bound: 1.0388447
time: 5.50 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0366378, upper bound: 1.0388429
time: 6.37 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -6.2020330, -1.9591279, -6.1842766, -2.0179734, -3.1840553, 3.2431502
1: -12.2414646, -8.9769354, -12.2208090, -8.9975929, -2.8290062, 2.8499913
2: -5.6216569, -2.2080774, -5.6145725, -2.2567019, -2.7764816, 2.7670641
3: -5.3784490, -1.5029228, -5.3525019, -1.5134199, -3.5520716, 3.5244904
4: -11.5405006, -7.5641451, -11.5071201, -7.5766025, -3.2317324, 3.3186927
5: -6.3009987, -3.0896358, -6.2814307, -3.1031904, -2.6441793, 2.6644068
6: -12.4368258, -8.6995010, -12.4200735, -8.7086525, -2.9246001, 2.9279957
7: -8.1801319, -4.6730866, -8.1508942, -4.6756282, -3.5045037, 3.4778075
8: 7.7354970, 10.0657616, 7.7506080, 10.0528030, -2.2244682, 2.2153537
9: -6.3482046, -2.7927980, -6.3363304, -2.8155026, -2.8603330, 2.9372563

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0387243, upper bound: 1.0391460
time: 5.46 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400754, upper bound: 1.0391462
time: 5.65 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -6.1998591, -1.9569540, -6.1713524, -2.0445619, -3.1623788, 3.2056751
1: -12.2408047, -8.9748821, -12.2015266, -8.9949598, -2.8281527, 2.8160434
2: -5.6126127, -2.2001805, -5.5637465, -2.2480221, -2.7598968, 2.7733846
3: -5.3757429, -1.5028479, -5.3126183, -1.5175371, -3.5441341, 3.4983773
4: -11.5447283, -7.5856366, -11.5084829, -7.6976881, -3.2248163, 3.2488279
5: -6.3007956, -3.0877991, -6.2785158, -3.1275911, -2.6220083, 2.6542764
6: -12.4359026, -8.6944675, -12.4118347, -8.7015896, -2.9300423, 2.9318452
7: -8.1852131, -4.6746626, -8.1549635, -4.6843429, -3.5008702, 3.4803009
8: 7.7322526, 10.0636921, 7.7501030, 10.0356674, -2.2199955, 2.2112808
9: -6.3491940, -2.8044033, -6.3348703, -2.8942525, -2.8384995, 2.8972316

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 540

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0371241, upper bound: 1.0389520
time: 11.12 seconds

## Relational analysis of IS_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0371241, upper bound: 1.0402581
time: 8.53 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -6.2037172, -1.9477458, -6.1948786, -1.9940381, -3.1924653, 3.2667828
1: -12.2469501, -8.9744158, -12.2345104, -8.9912529, -2.8449011, 2.8662524
2: -5.6229410, -2.1984062, -5.6206608, -2.2354560, -2.7971497, 2.7853017
3: -5.3863239, -1.5006950, -5.3708830, -1.5042117, -3.5699224, 3.5381145
4: -11.5466614, -7.5620060, -11.5237951, -7.5678282, -3.2483215, 3.3357997
5: -6.3025012, -3.0794802, -6.2900915, -3.0817943, -2.6533060, 2.6848907
6: -12.4390497, -8.6935787, -12.4278774, -8.6957169, -2.9401927, 2.9422126
7: -8.1877193, -4.6719465, -8.1703491, -4.6722760, -3.5154433, 3.4984026
8: 7.7302885, 10.0681171, 7.7388449, 10.0601463, -2.2434139, 2.2300947
9: -6.3509579, -2.7878113, -6.3480172, -2.8028529, -2.8725677, 2.9539509

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392316, upper bound: 1.0405708
time: 5.91 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405681, upper bound: 1.0405670
time: 5.17 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.75 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 25.75
Output dim: 8, lower bound: -1.0389547, upper bound: 1.0348681
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 25.75
Output dim: 8, lower bound: -1.0392456, upper bound: 1.0382958
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 25.75
Output dim: 8, lower bound: -1.0389547, upper bound: 1.0348681
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 25.75
Output dim: 8, lower bound: -1.0392480, upper bound: 1.0382958
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 25.75
Output dim: 8, lower bound: -1.0393655, upper bound: 1.0362236
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 25.75
Output dim: 8, lower bound: -1.0396775, upper bound: 1.0396672
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 25.75
Output dim: 8, lower bound: -1.0393655, upper bound: 1.0362215
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 25.75
Output dim: 8, lower bound: -1.0396775, upper bound: 1.0396676
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 25.75
Output dim: 8, lower bound: -1.0353390, upper bound: 1.0388447
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 25.75
Output dim: 8, lower bound: -1.0366378, upper bound: 1.0388429
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 25.75
Output dim: 8, lower bound: -1.0387243, upper bound: 1.0391460
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 25.75
Output dim: 8, lower bound: -1.0400754, upper bound: 1.0391462
IS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 25.75
Output dim: 8, lower bound: -1.0371241, upper bound: 1.0389520
IS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 25.75
Output dim: 8, lower bound: -1.0371241, upper bound: 1.0402581
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 25.75
Output dim: 8, lower bound: -1.0392316, upper bound: 1.0405708
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 25.75
Output dim: 8, lower bound: -1.0405681, upper bound: 1.0405670

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -6.1474953, -2.0566833, -6.1582537, -2.0279157, -3.1522732, 3.1313086
1: -12.1807365, -8.9980106, -12.1994944, -8.9986305, -2.7699237, 2.7804890
2: -5.5439463, -2.2613964, -5.5857368, -2.2622876, -2.7015228, 2.7259226
3: -5.3028522, -1.5234342, -5.3401003, -1.5192556, -3.4691677, 3.5040245
4: -11.4975004, -7.7102976, -11.5002699, -7.6107073, -3.2143879, 3.1676702
5: -6.2747726, -3.1411271, -6.2774591, -3.1148691, -2.6169119, 2.5938540
6: -12.4078264, -8.7094936, -12.4151230, -8.7115355, -2.8889265, 2.8958812
7: -8.1421547, -4.6923418, -8.1430702, -4.6852069, -3.4569478, 3.4507284
8: 7.7578478, 10.0282173, 7.7551379, 10.0433588, -2.1755772, 2.1717353
9: -6.3286524, -2.9015927, -6.3311033, -2.8342900, -2.8628473, 2.8147326

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0389528, upper bound: 1.0335737
time: 9.17 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0389529, upper bound: 1.0348727
time: 6.80 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -6.1710367, -2.0061817, -6.1621284, -2.0187166, -3.1839476, 3.1614475
1: -12.2138605, -8.9943275, -12.2056417, -8.9981718, -2.8202410, 2.7972622
2: -5.6008606, -2.2489302, -5.5960608, -2.2605476, -2.7137084, 2.7661934
3: -5.3610687, -1.5101225, -5.3506393, -1.5171015, -3.5089064, 3.5257521
4: -11.5127783, -7.5804315, -11.5021973, -7.5870790, -3.3020697, 3.1911631
5: -6.2863016, -3.0952940, -6.2791467, -3.1065421, -2.6380949, 2.6251369
6: -12.4237862, -8.7036333, -12.4182100, -8.7106514, -2.8993034, 2.9059186
7: -8.1574068, -4.6802783, -8.1455431, -4.6824942, -3.4749126, 3.4652648
8: 7.7466345, 10.0527372, 7.7531829, 10.0477810, -2.1943831, 2.1951492
9: -6.3417888, -2.8101788, -6.3328576, -2.8177021, -2.9080877, 2.8488154

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392437, upper bound: 1.0369561
time: 13.98 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392438, upper bound: 1.0383033
time: 6.53 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -6.1474953, -2.0566833, -6.1892638, -1.9808357, -3.1627150, 3.1616292
1: -12.1807365, -8.9980106, -12.2271147, -8.9812326, -2.7881365, 2.8089733
2: -5.5439463, -2.2613964, -5.6065006, -2.2214775, -2.7340293, 2.7372360
3: -5.3028522, -1.5234342, -5.3575258, -1.5120506, -3.4775400, 3.5177808
4: -11.4975004, -7.7102976, -11.5280666, -7.5944343, -3.2252207, 3.1949363
5: -6.2747726, -3.1411271, -6.2921343, -3.1092048, -2.6230764, 2.6083808
6: -12.4078264, -8.7094936, -12.4280987, -8.7073975, -2.8946805, 2.9075255
7: -8.1421547, -4.6923418, -8.1658354, -4.6780167, -3.4641380, 3.4734936
8: 7.7578478, 10.0282173, 7.7439861, 10.0563726, -2.1895885, 2.1834795
9: -6.3286524, -2.9015927, -6.3375163, -2.8167863, -2.8759050, 2.8207941

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0397710, upper bound: 1.0335668
time: 5.75 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0397709, upper bound: 1.0348646
time: 6.22 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -6.1710367, -2.0061817, -6.1931095, -1.9716358, -3.2237730, 3.1917458
1: -12.2138605, -8.9943275, -12.2332678, -8.9807749, -2.8384390, 2.8256993
2: -5.6008606, -2.2489302, -5.6168265, -2.2197113, -2.7459421, 2.7878547
3: -5.3610687, -1.5101225, -5.3681040, -1.5099022, -3.5172768, 3.5412407
4: -11.5127783, -7.5804315, -11.5300026, -7.5708055, -3.3186865, 3.2184534
5: -6.2863016, -3.0952940, -6.2938461, -3.1008787, -2.6442575, 2.6397057
6: -12.4237862, -8.7036333, -12.4312420, -8.7065058, -2.9050531, 2.9176054
7: -8.1574068, -4.6802783, -8.1683264, -4.6752987, -3.4821081, 3.4880481
8: 7.7466345, 10.0527372, 7.7420254, 10.0607967, -2.2083640, 2.2068632
9: -6.3417888, -2.8101788, -6.3392596, -2.8001897, -2.9254856, 2.8548646

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400755, upper bound: 1.0369454
time: 6.37 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400757, upper bound: 1.0382910
time: 5.29 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -6.1491609, -2.0453010, -6.1688375, -2.0039868, -3.1605940, 3.1560497
1: -12.1862755, -8.9955158, -12.2132387, -8.9922819, -2.7859230, 2.7967334
2: -5.5452175, -2.2517624, -5.5918112, -2.2410307, -2.7221622, 2.7441673
3: -5.3107920, -1.5212209, -5.3584719, -1.5100567, -3.4870596, 3.5176711
4: -11.5037012, -7.7081833, -11.5170450, -7.6019373, -3.2305279, 3.1849065
5: -6.2762847, -3.1309214, -6.2861133, -3.0934310, -2.6260881, 2.6143961
6: -12.4100380, -8.7035513, -12.4229240, -8.6985779, -2.9044819, 2.9101439
7: -8.1497450, -4.6912065, -8.1625500, -4.6818557, -3.4678893, 3.4713435
8: 7.7526402, 10.0305748, 7.7433782, 10.0506744, -2.1944809, 2.1864436
9: -6.3313632, -2.8964870, -6.3427610, -2.8216610, -2.8748040, 2.8315158

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 540

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0393637, upper bound: 1.0349313
time: 9.97 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0393637, upper bound: 1.0362284
time: 6.19 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -6.1727223, -1.9947748, -6.1727219, -1.9947796, -3.1923084, 3.1862059
1: -12.2193823, -8.9918165, -12.2193813, -8.9918156, -2.8361468, 2.8135104
2: -5.6021328, -2.2392836, -5.6021366, -2.2392843, -2.7343779, 2.7834024
3: -5.3690147, -1.5078959, -5.3690143, -1.5078967, -3.5268154, 3.5394197
4: -11.5189676, -7.5783119, -11.5189676, -7.5783076, -3.3187237, 3.2084012
5: -6.2878003, -3.0851107, -6.2878041, -3.0851116, -2.6472497, 2.6456518
6: -12.4260101, -8.6976948, -12.4260111, -8.6976995, -2.9148521, 2.9201717
7: -8.1650400, -4.6791420, -8.1650362, -4.6791410, -3.4858990, 3.4858942
8: 7.7414160, 10.0551033, 7.7414179, 10.0551033, -2.2133031, 2.2099411
9: -6.3445272, -2.8050778, -6.3445263, -2.8050733, -2.9202852, 2.8656015

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396756, upper bound: 1.0383445
time: 10.12 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396756, upper bound: 1.0396746
time: 5.90 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -6.1491609, -2.0453010, -6.1998568, -1.9569564, -3.1701245, 3.1863847
1: -12.1862755, -8.9955158, -12.2407999, -8.9748821, -2.8041549, 2.8251815
2: -5.5452175, -2.2517624, -5.6126146, -2.2001834, -2.7539964, 2.7555175
3: -5.3107920, -1.5212209, -5.3757410, -1.5028484, -3.4954386, 3.5310402
4: -11.5037012, -7.7081833, -11.5447273, -7.5856376, -3.2413244, 3.2120204
5: -6.2762847, -3.1309214, -6.3007956, -3.0878038, -2.6321955, 2.6289644
6: -12.4100380, -8.7035513, -12.4359026, -8.6944704, -2.9101481, 2.9218488
7: -8.1497450, -4.6912065, -8.1852121, -4.6746635, -3.4750814, 3.4940057
8: 7.7526402, 10.0305748, 7.7322555, 10.0636911, -2.2085485, 2.1981218
9: -6.3313632, -2.8964870, -6.3491936, -2.8044064, -2.8875504, 2.8376074

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 540

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0402581, upper bound: 1.0349222
time: 8.60 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0402604, upper bound: 1.0362217
time: 6.67 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -6.1727223, -1.9947748, -6.2037163, -1.9477482, -3.2312555, 3.2165236
1: -12.2193823, -8.9918165, -12.2469482, -8.9744177, -2.8543653, 2.8419232
2: -5.6021328, -2.2392836, -5.6229410, -2.1984084, -2.7659192, 2.8051286
3: -5.3690147, -1.5078959, -5.3863225, -1.5006955, -3.5351877, 3.5547452
4: -11.5189676, -7.5783119, -11.5466585, -7.5620060, -3.3353271, 3.2355361
5: -6.2878003, -3.0851107, -6.3025031, -3.0794840, -2.6533556, 2.6602631
6: -12.4260101, -8.6976948, -12.4390488, -8.6935835, -2.9205189, 2.9319205
7: -8.1650400, -4.6791420, -8.1877155, -4.6719475, -3.4930925, 3.5085735
8: 7.7414160, 10.0551033, 7.7302909, 10.0681171, -2.2273397, 2.2215893
9: -6.3445272, -2.8050778, -6.3509579, -2.7878122, -2.9374151, 2.8716803

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405683, upper bound: 1.0383315
time: 6.04 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405685, upper bound: 1.0396645
time: 5.88 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -6.1581669, -1.9925830, -6.1518707, -2.0801435, -3.0782738, 3.1561487
1: -12.2053032, -8.9942732, -12.1752129, -9.0027580, -2.7773218, 2.7882862
2: -5.5856915, -2.2336373, -5.5558381, -2.2774599, -2.7145977, 2.7633452
3: -5.3401442, -1.5329211, -5.2918110, -1.5392673, -3.4981647, 3.4447823
4: -11.5021868, -7.6101327, -11.4849596, -7.7168355, -2.9897480, 3.2085891
5: -6.2310233, -3.1434846, -6.2365956, -3.1515961, -2.5343657, 2.5277591
6: -12.3996410, -8.7333412, -12.3885393, -8.7175884, -2.8697968, 2.8368464
7: -8.1510229, -4.6899686, -8.1265249, -4.6893015, -3.4180250, 3.4365563
8: 7.7493887, 10.0405102, 7.7649031, 10.0238943, -2.1612606, 2.1681337
9: -6.3038244, -2.8428473, -6.3156657, -2.9244645, -2.7601337, 2.8312640

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 5749

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0344119, upper bound: 1.0388434
time: 9.34 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0344119, upper bound: 1.0388436
time: 7.39 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -6.1981654, -1.9683504, -6.1607814, -2.0684500, -3.1539626, 3.1652033
1: -12.2352982, -8.9773979, -12.1877823, -9.0012579, -2.7900405, 2.7997432
2: -5.6113272, -2.2098603, -5.5576653, -2.2692423, -2.7261858, 2.7403278
3: -5.3678646, -1.5050938, -5.2942610, -1.5267258, -3.5106993, 3.4704242
4: -11.5385580, -7.5877948, -11.4917698, -7.7064691, -3.2033100, 3.2133727
5: -6.2992330, -3.0979633, -6.2697945, -3.1490269, -2.5862627, 2.5990095
6: -12.4336538, -8.7003927, -12.4040346, -8.7145443, -2.9015064, 2.9021378
7: -8.1776190, -4.6758041, -8.1355782, -4.6876974, -3.4899216, 3.4597740
8: 7.7374649, 10.0613289, 7.7618346, 10.0283451, -2.1981773, 2.1964436
9: -6.3464441, -2.8094268, -6.3232265, -2.9069335, -2.8262243, 2.8523674

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0357107, upper bound: 1.0388433
time: 5.99 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0357107, upper bound: 1.0388456
time: 6.36 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -6.1620359, -1.9833841, -6.1753893, -2.0296655, -3.1082306, 3.2167840
1: -12.2113972, -8.9938173, -12.2081070, -8.9991131, -2.7940893, 2.8387756
2: -5.5960226, -2.2318540, -5.6127481, -2.2649169, -2.7488308, 2.7750804
3: -5.3506827, -1.5307853, -5.3500166, -1.5259709, -3.5209341, 3.4854488
4: -11.5041676, -7.5864868, -11.5004616, -7.5869231, -3.0218363, 3.2955346
5: -6.2327452, -3.1351614, -6.2482128, -3.1057940, -2.5657134, 2.5661035
6: -12.4028015, -8.7324409, -12.4046288, -8.7116661, -2.8799610, 2.8470192
7: -8.1534472, -4.6872702, -8.1417837, -4.6772399, -3.4490881, 3.4545135
8: 7.7474189, 10.0449352, 7.7536583, 10.0483332, -2.1830106, 2.1870141
9: -6.3056188, -2.8262451, -6.3288779, -2.8330429, -2.7942595, 2.8888674

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0377950, upper bound: 1.0391461
time: 5.57 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0377950, upper bound: 1.0391459
time: 5.25 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -6.2020178, -1.9591465, -6.1842690, -2.0179849, -3.1840343, 3.2262442
1: -12.2414503, -8.9769373, -12.2207985, -8.9975939, -2.8066969, 2.8499794
2: -5.6216545, -2.2080874, -5.6145701, -2.2567079, -2.7829533, 2.7522292
3: -5.3784451, -1.5029411, -5.3525004, -1.5134325, -3.5503101, 3.5098906
4: -11.5404902, -7.5641680, -11.5071163, -7.5766168, -3.2267694, 3.3036971
5: -6.3009443, -3.0896378, -6.2814007, -3.1031923, -2.6175985, 2.6457114
6: -12.4367990, -8.6995029, -12.4200621, -8.7086535, -2.9116278, 2.9122028
7: -8.1801195, -4.6730871, -8.1508856, -4.6756306, -3.5044889, 3.4777985
8: 7.7355022, 10.0657539, 7.7506094, 10.0528002, -2.2215266, 2.2151284
9: -6.3481932, -2.7928326, -6.3363218, -2.8155220, -2.8602991, 2.9226122

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391460, upper bound: 1.0391457
time: 5.21 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391461, upper bound: 1.0391457
time: 5.30 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.06 seconds
IS_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0389528, upper bound: 1.0335737
IS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0389529, upper bound: 1.0348727
IS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0392437, upper bound: 1.0369561
IS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0392438, upper bound: 1.0383033
IS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0397710, upper bound: 1.0335668
IS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0397709, upper bound: 1.0348646
IS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0400755, upper bound: 1.0369454
IS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0400757, upper bound: 1.0382910
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0393637, upper bound: 1.0349313
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0393637, upper bound: 1.0362284
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0396756, upper bound: 1.0383445
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0396756, upper bound: 1.0396746
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0402581, upper bound: 1.0349222
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0402604, upper bound: 1.0362217
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0405683, upper bound: 1.0383315
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0405685, upper bound: 1.0396645
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0344119, upper bound: 1.0388434
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0344119, upper bound: 1.0388436
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0357107, upper bound: 1.0388433
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0357107, upper bound: 1.0388456
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0377950, upper bound: 1.0391461
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0377950, upper bound: 1.0391459
IS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0391460, upper bound: 1.0391457
IS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 25.06
Output dim: 8, lower bound: -1.0391461, upper bound: 1.0391457
IS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 8, lower bound: -1.0371241, upper bound: 1.0389520
IS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 8, lower bound: -1.0371241, upper bound: 1.0402581
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 8, lower bound: -1.0392316, upper bound: 1.0405708
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 8, lower bound: -1.0405681, upper bound: 1.0405670
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.225489616394043
rel_dist={8: [-1.040580051898944, 1.0405791313826693]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2420.10 seconds
