## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 1.84108648411
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464)
1: (-0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231)
2: (-0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522)
3: (-0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149)
4: (-0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112)
5: (-0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641)
6: (-0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580)
7: (-0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284)
8: (-0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284)
9: (-0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595)

## BASE Result
execution time: IAR + LP analysis = 1.29 + 2.91 = 4.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -1.9162456, upper bound: 1.9162456


# Binary Search by BASE starts (time budget: 2695.80 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.094357967376709
rel_dist={6: [-1.9135725282356488, 1.9135718444666665]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.094357967376709
rel_dist={6: [-1.9113006693296792, 1.9113006693296786]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.094357967376709
rel_dist={6: [-1.9066430217684205, 1.9066430217684207]}

## Binary Search Result
Binary search time: 16.67 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2679.12 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 189

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9092793, upper bound: 1.8660586
time: 1.79 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877
time: 2.80 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 4.70 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 4.70
Output dim: 6, lower bound: -1.9092793, upper bound: 1.8660586
IS_A2, status: Status.UNKNOWN, split count: 1, time: 4.70
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.4283954, 1.2709696, -0.4769118, 1.3269346, -1.7553300, 1.7478814
1: -0.5065504, 0.5120057, -0.5433673, 0.5461558, -1.0527062, 1.0553730
2: -0.5936686, 0.6343641, -0.6294971, 0.6858552, -1.2795238, 1.2638612
3: -0.4099944, 0.4626325, -0.4428055, 0.5047094, -0.9147038, 0.9054380
4: -0.5260350, 0.5839709, -0.5623835, 0.6495277, -1.1755627, 1.1463544
5: -0.6253715, 0.7425824, -0.6725667, 0.7972974, -1.4226689, 1.4151490
6: -0.5344308, 1.4685735, -0.6051830, 1.4891748, -2.0236056, 2.0737565
7: -0.5838713, 0.6439485, -0.6342677, 0.6814606, -1.2653320, 1.2782162
8: -0.5708863, 0.6576684, -0.6106737, 0.7169547, -1.2878411, 1.2683420
9: -0.4413340, 0.4900914, -0.4836978, 0.5420617, -0.9833957, 0.9737892

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 189

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877
time: 2.08 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877
time: 2.54 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.1578722, 2.0902753, -0.4592743, 1.3080159, -2.4658880, 2.5495496
1: -1.0186968, 0.9981120, -0.5311626, 0.5342227, -1.5529195, 1.5292746
2: -1.0924278, 1.3473158, -0.6175320, 0.6685661, -1.7609940, 1.9648478
3: -0.8712537, 1.0596817, -0.4319794, 0.4902656, -1.3615193, 1.4916611
4: -1.0514562, 1.4630030, -0.5498180, 0.6282432, -1.6796994, 2.0128212
5: -1.3404843, 1.4613965, -0.6556945, 0.7794052, -2.1198895, 2.1170909
6: -1.5363673, 1.8078189, -0.5814891, 1.4815513, -3.0179186, 2.3893080
7: -1.2797452, 1.1851227, -0.6174742, 0.6684646, -1.9482098, 1.8025969
8: -1.1523812, 1.4595408, -0.5971164, 0.6973907, -1.8497719, 2.0566573
9: -1.0169351, 1.1958456, -0.4699131, 0.5248307, -1.5417658, 1.6657586

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 189

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877
time: 2.57 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877
time: 2.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 6.73 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 6.73
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 6.73
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 6.73
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 6.73
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.4283954, 1.2709696, -0.4283954, 1.2709696, -1.6993650, 1.6993650
1: -0.5065504, 0.5120057, -0.5065504, 0.5120057, -1.0185561, 1.0185561
2: -0.5936686, 0.6343641, -0.5936686, 0.6343641, -1.2280327, 1.2280327
3: -0.4099944, 0.4626325, -0.4099944, 0.4626325, -0.8726270, 0.8726270
4: -0.5260350, 0.5839709, -0.5260350, 0.5839709, -1.1100059, 1.1100059
5: -0.6253715, 0.7425824, -0.6253715, 0.7425824, -1.3679538, 1.3679538
6: -0.5344308, 1.4685735, -0.5344308, 1.4685735, -2.0030043, 2.0030043
7: -0.5838713, 0.6439485, -0.5838713, 0.6439485, -1.2278198, 1.2278198
8: -0.5708863, 0.6576684, -0.5708863, 0.6576684, -1.2285547, 1.2285547
9: -0.4413340, 0.4900914, -0.4413340, 0.4900914, -0.9314255, 0.9314255

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8900152, upper bound: 1.8319765
time: 2.10 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8843870, upper bound: 1.8319511
time: 2.30 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.4283954, 1.2709696, -1.1578722, 2.0902753, -2.5186706, 2.4288418
1: -0.5065504, 0.5120057, -1.0186968, 0.9981120, -1.5046624, 1.5307025
2: -0.5936686, 0.6343641, -1.0924278, 1.3473158, -1.9409844, 1.7267920
3: -0.4099944, 0.4626325, -0.8712537, 1.0596817, -1.4696760, 1.3338861
4: -0.5260350, 0.5839709, -1.0514562, 1.4630030, -1.9890380, 1.6354271
5: -0.6253715, 0.7425824, -1.3404843, 1.4613965, -2.0867679, 2.0830667
6: -0.5344308, 1.4685735, -1.5363673, 1.8078189, -2.3422496, 3.0049407
7: -0.5838713, 0.6439485, -1.2797452, 1.1851227, -1.7689941, 1.9236937
8: -0.5708863, 0.6576684, -1.1523812, 1.4595408, -2.0304272, 1.8100495
9: -0.4413340, 0.4900914, -1.0169351, 1.1958456, -1.6371796, 1.5070266

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8900152, upper bound: 1.8319765
time: 2.09 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8843870, upper bound: 1.8319511
time: 2.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -1.1578722, 2.0902753, -0.4283954, 1.2709696, -2.4288418, 2.5186706
1: -1.0186968, 0.9981120, -0.5065504, 0.5120057, -1.5307025, 1.5046624
2: -1.0924278, 1.3473158, -0.5936686, 0.6343641, -1.7267920, 1.9409844
3: -0.8712537, 1.0596817, -0.4099944, 0.4626325, -1.3338861, 1.4696760
4: -1.0514562, 1.4630030, -0.5260350, 0.5839709, -1.6354271, 1.9890380
5: -1.3404843, 1.4613965, -0.6253715, 0.7425824, -2.0830667, 2.0867679
6: -1.5363673, 1.8078189, -0.5344308, 1.4685735, -3.0049407, 2.3422496
7: -1.2797452, 1.1851227, -0.5838713, 0.6439485, -1.9236937, 1.7689941
8: -1.1523812, 1.4595408, -0.5708863, 0.6576684, -1.8100495, 2.0304272
9: -1.0169351, 1.1958456, -0.4413340, 0.4900914, -1.5070266, 1.6371796

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8638037, upper bound: 1.8648489
time: 2.06 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877
time: 2.19 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -1.1578722, 2.0902753, -1.1578722, 2.0902753, -3.2481475, 3.2481475
1: -1.0186968, 0.9981120, -1.0186968, 0.9981120, -2.0168087, 2.0168087
2: -1.0924278, 1.3473158, -1.0924278, 1.3473158, -2.4397435, 2.4397435
3: -0.8712537, 1.0596817, -0.8712537, 1.0596817, -1.9309354, 1.9309354
4: -1.0514562, 1.4630030, -1.0514562, 1.4630030, -2.5144591, 2.5144591
5: -1.3404843, 1.4613965, -1.3404843, 1.4613965, -2.8018808, 2.8018808
6: -1.5363673, 1.8078189, -1.5363673, 1.8078189, -3.3441863, 3.3441863
7: -1.2797452, 1.1851227, -1.2797452, 1.1851227, -2.4648681, 2.4648681
8: -1.1523812, 1.4595408, -1.1523812, 1.4595408, -2.6119220, 2.6119220
9: -1.0169351, 1.1958456, -1.0169351, 1.1958456, -2.2127807, 2.2127807

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8638037, upper bound: 1.8648489
time: 2.82 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877
time: 2.22 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.44 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.44
Output dim: 6, lower bound: -1.8900152, upper bound: 1.8319765
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.44
Output dim: 6, lower bound: -1.8843870, upper bound: 1.8319511
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.44
Output dim: 6, lower bound: -1.8900152, upper bound: 1.8319765
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.44
Output dim: 6, lower bound: -1.8843870, upper bound: 1.8319511
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.44
Output dim: 6, lower bound: -1.8638037, upper bound: 1.8648489
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.44
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.44
Output dim: 6, lower bound: -1.8638037, upper bound: 1.8648489
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.44
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2336920, 0.9276679, -0.4283954, 1.2709696, -1.5046617, 1.3560632
1: -0.3219900, 0.3554728, -0.5065504, 0.5120057, -0.8339957, 0.8620232
2: -0.4061271, 0.4243887, -0.5936686, 0.6343641, -1.0404912, 1.0180573
3: -0.2861868, 0.2612684, -0.4099944, 0.4626325, -0.7488193, 0.6712628
4: -0.3218213, 0.3896247, -0.5260350, 0.5839709, -0.9057922, 0.9156598
5: -0.4451328, 0.5103453, -0.6253715, 0.7425824, -1.1877152, 1.1357167
6: -0.1471975, 1.2969497, -0.5344308, 1.4685735, -1.6157711, 1.8313806
7: -0.3616745, 0.4618227, -0.5838713, 0.6439485, -1.0056230, 1.0456941
8: -0.3729174, 0.4232192, -0.5708863, 0.6576684, -1.0305858, 0.9941055
9: -0.2493206, 0.2945979, -0.4413340, 0.4900914, -0.7394120, 0.7359320

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8905455, upper bound: 1.8905455
time: 2.19 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8905455, upper bound: 1.8905455
time: 2.12 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2360516, 0.9213381, -0.3830998, 1.1939206, -1.4299722, 1.3044379
1: -0.3221140, 0.3588542, -0.4624637, 0.4786091, -0.8007231, 0.8213180
2: -0.4051805, 0.4282180, -0.5521010, 0.5773075, -0.9824880, 0.9803190
3: -0.2861544, 0.2650071, -0.3769137, 0.4171165, -0.7032709, 0.6419208
4: -0.3222730, 0.3931868, -0.4822236, 0.5206205, -0.8428935, 0.8754104
5: -0.4428855, 0.5166113, -0.5755689, 0.6879572, -1.1308427, 1.0921803
6: -0.1440104, 1.2948549, -0.4452221, 1.4325594, -1.5765698, 1.7400770
7: -0.3651637, 0.4619819, -0.5307168, 0.6035373, -0.9687010, 0.9926987
8: -0.3766791, 0.4280382, -0.5228423, 0.5970218, -0.9737008, 0.9508805
9: -0.2533768, 0.3006897, -0.3955613, 0.4396504, -0.6930272, 0.6962509

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8900268, upper bound: 1.8897895
time: 2.95 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8905455, upper bound: 1.8905455
time: 2.05 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2336920, 0.9276679, -1.1578722, 2.0902753, -2.3239672, 2.0855401
1: -0.3219900, 0.3554728, -1.0186968, 0.9981120, -1.3201020, 1.3741696
2: -0.4061271, 0.4243887, -1.0924278, 1.3473158, -1.7534429, 1.5168166
3: -0.2861868, 0.2612684, -0.8712537, 1.0596817, -1.3458683, 1.1325221
4: -0.3218213, 0.3896247, -1.0514562, 1.4630030, -1.7848244, 1.4410809
5: -0.4451328, 0.5103453, -1.3404843, 1.4613965, -1.9065293, 1.8508296
6: -0.1471975, 1.2969497, -1.5363673, 1.8078189, -1.9550164, 2.8333170
7: -0.3616745, 0.4618227, -1.2797452, 1.1851227, -1.5467973, 1.7415680
8: -0.3729174, 0.4232192, -1.1523812, 1.4595408, -1.8324583, 1.5756004
9: -0.2493206, 0.2945979, -1.0169351, 1.1958456, -1.4451662, 1.3115330

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 122

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8248847
time: 2.29 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8900152, upper bound: 1.8319765
time: 2.25 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2360516, 0.9213381, -1.0940137, 2.0133772, -2.2494287, 2.0153518
1: -0.3221140, 0.3588542, -0.9736349, 0.9566177, -1.2787317, 1.3324891
2: -0.4051805, 0.4282180, -1.0481801, 1.2863832, -1.6915636, 1.4763981
3: -0.2861544, 0.2650071, -0.8298265, 1.0085218, -1.2946762, 1.0948336
4: -0.3222730, 0.3931868, -1.0049838, 1.3891842, -1.7114573, 1.3981706
5: -0.4428855, 0.5166113, -1.2768829, 1.3999461, -1.8428316, 1.7934942
6: -0.1440104, 1.2948549, -1.4457482, 1.7707210, -1.9147314, 2.7406030
7: -0.3651637, 0.4619819, -1.2203215, 1.1372957, -1.5024594, 1.6823034
8: -0.3766791, 0.4280382, -1.1012522, 1.3909206, -1.7675997, 1.5292903
9: -0.2533768, 0.3006897, -0.9687698, 1.1362712, -1.3896481, 1.2694595

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 122

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8774045, upper bound: 1.8248551
time: 2.23 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8843870, upper bound: 1.8319511
time: 2.30 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.1571229, 2.1073875, -0.4067150, 1.2407022, -2.3978250, 2.5141025
1: -1.0190454, 0.9931424, -0.4869262, 0.4960527, -1.5150981, 1.4800687
2: -1.0939938, 1.3421152, -0.5755167, 0.6072444, -1.7012382, 1.9176319
3: -0.8575323, 1.0565397, -0.3936818, 0.4409779, -1.2985102, 1.4502214
4: -1.0527849, 1.4547693, -0.5070575, 0.5511866, -1.6039715, 1.9618268
5: -1.3438350, 1.4506361, -0.6038042, 0.7142713, -2.0581064, 2.0544405
6: -1.5473984, 1.8286030, -0.4965411, 1.4581203, -3.0055189, 2.3251443
7: -1.2758220, 1.1852087, -0.5581359, 0.6260951, -1.9019171, 1.7433445
8: -1.1452086, 1.4524922, -0.5503402, 0.6266192, -1.7718277, 2.0028324
9: -1.0130838, 1.1884294, -0.4185991, 0.4651394, -1.4782233, 1.6070285

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8248847, upper bound: 1.8822187
time: 2.19 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8248551, upper bound: 1.8774045
time: 2.14 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.1064560, 2.0360427, -0.4283954, 1.2709696, -2.3774257, 2.4644380
1: -0.9828035, 0.9629667, -0.5065504, 0.5120057, -1.4948092, 1.4695172
2: -1.0578785, 1.2963203, -0.5936686, 0.6343641, -1.6922426, 1.8899889
3: -0.8345226, 1.0170850, -0.4099944, 0.4626325, -1.2971551, 1.4270794
4: -1.0146995, 1.3996853, -0.5260350, 0.5839709, -1.5986704, 1.9257202
5: -1.2906734, 1.4091601, -0.6253715, 0.7425824, -2.0332558, 2.0345316
6: -1.4684881, 1.7880827, -0.5344308, 1.4685735, -2.9370615, 2.3225136
7: -1.2301536, 1.1471224, -0.5838713, 0.6439485, -1.8741021, 1.7309937
8: -1.1096203, 1.4020764, -0.5708863, 0.6576684, -1.7672887, 1.9729626
9: -0.9756740, 1.1447871, -0.4413340, 0.4900914, -1.4657654, 1.5861211

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8319765, upper bound: 1.8900152
time: 1.97 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8319511, upper bound: 1.8843870
time: 2.31 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.1571229, 2.1073875, -1.1282098, 2.0590510, -3.2161739, 3.2355974
1: -1.0190454, 0.9931424, -0.9979855, 0.9777729, -1.9968183, 1.9911280
2: -1.0939938, 1.3421152, -1.0724709, 1.3178763, -2.4118700, 2.4145861
3: -0.8575323, 1.0565397, -0.8495320, 1.0351305, -1.8926628, 1.9060717
4: -1.0527849, 1.4547693, -1.0302776, 1.4265454, -2.4793303, 2.4850469
5: -1.3438350, 1.4506361, -1.3118010, 1.4308927, -2.7747278, 2.7624371
6: -1.5473984, 1.8286030, -1.4972399, 1.7963924, -3.3437910, 3.3258429
7: -1.2758220, 1.1852087, -1.2511563, 1.1631663, -2.4389882, 2.4363651
8: -1.1452086, 1.4524922, -1.1274332, 1.4262674, -2.5714760, 2.5799255
9: -1.0130838, 1.1884294, -0.9932972, 1.1663489, -2.1794329, 2.1817265

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 122

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8241928, upper bound: 1.8459148
time: 1.79 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8241023, upper bound: 1.8303278
time: 2.45 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.1064560, 2.0360427, -1.1578722, 2.0902753, -3.1967313, 3.1939149
1: -0.9828035, 0.9629667, -1.0186968, 0.9981120, -1.9809154, 1.9816635
2: -1.0578785, 1.2963203, -1.0924278, 1.3473158, -2.4051943, 2.3887482
3: -0.8345226, 1.0170850, -0.8712537, 1.0596817, -1.8942043, 1.8883386
4: -1.0146995, 1.3996853, -1.0514562, 1.4630030, -2.4777026, 2.4511414
5: -1.2906734, 1.4091601, -1.3404843, 1.4613965, -2.7520700, 2.7496443
6: -1.4684881, 1.7880827, -1.5363673, 1.8078189, -3.2763071, 3.3244500
7: -1.2301536, 1.1471224, -1.2797452, 1.1851227, -2.4152763, 2.4268675
8: -1.1096203, 1.4020764, -1.1523812, 1.4595408, -2.5691612, 2.5544577
9: -0.9756740, 1.1447871, -1.0169351, 1.1958456, -2.1715195, 2.1617222

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 122

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8648489, upper bound: 1.8638037
time: 2.24 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8648489, upper bound: 1.8654877
time: 1.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.45 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 6, lower bound: -1.8905455, upper bound: 1.8905455
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 6, lower bound: -1.8905455, upper bound: 1.8905455
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 6, lower bound: -1.8900268, upper bound: 1.8897895
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 6, lower bound: -1.8905455, upper bound: 1.8905455
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8248847
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 6, lower bound: -1.8900152, upper bound: 1.8319765
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 6, lower bound: -1.8774045, upper bound: 1.8248551
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 6, lower bound: -1.8843870, upper bound: 1.8319511
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 6, lower bound: -1.8248847, upper bound: 1.8822187
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 6, lower bound: -1.8248551, upper bound: 1.8774045
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 6, lower bound: -1.8319765, upper bound: 1.8900152
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 6, lower bound: -1.8319511, upper bound: 1.8843870
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 6, lower bound: -1.8241928, upper bound: 1.8459148
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 5.45
Output dim: 6, lower bound: -1.8241023, upper bound: 1.8303278
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 6, lower bound: -1.8648489, upper bound: 1.8638037
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 6, lower bound: -1.8648489, upper bound: 1.8654877

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2336920, 0.9276679, -0.2336920, 0.9276679, -1.1613599, 1.1613599
1: -0.3219900, 0.3554728, -0.3219900, 0.3554728, -0.6774628, 0.6774628
2: -0.4061271, 0.4243887, -0.4061271, 0.4243887, -0.8305158, 0.8305158
3: -0.2861868, 0.2612684, -0.2861868, 0.2612684, -0.5474551, 0.5474551
4: -0.3218213, 0.3896247, -0.3218213, 0.3896247, -0.7114460, 0.7114460
5: -0.4451328, 0.5103453, -0.4451328, 0.5103453, -0.9554781, 0.9554781
6: -0.1471975, 1.2969497, -0.1471975, 1.2969497, -1.4441473, 1.4441473
7: -0.3616745, 0.4618227, -0.3616745, 0.4618227, -0.8234972, 0.8234972
8: -0.3729174, 0.4232192, -0.3729174, 0.4232192, -0.7961366, 0.7961366
9: -0.2493206, 0.2945979, -0.2493206, 0.2945979, -0.5439185, 0.5439185

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8991593, upper bound: 1.8901671
time: 2.24 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8991594, upper bound: 1.8905728
time: 2.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2336920, 0.9276679, -0.2360516, 0.9213381, -1.1550301, 1.1637194
1: -0.3219900, 0.3554728, -0.3221140, 0.3588542, -0.6808442, 0.6775868
2: -0.4061271, 0.4243887, -0.4051805, 0.4282180, -0.8343451, 0.8295692
3: -0.2861868, 0.2612684, -0.2861544, 0.2650071, -0.5511939, 0.5474228
4: -0.3218213, 0.3896247, -0.3222730, 0.3931868, -0.7150081, 0.7118977
5: -0.4451328, 0.5103453, -0.4428855, 0.5166113, -0.9617441, 0.9532307
6: -0.1471975, 1.2969497, -0.1440104, 1.2948549, -1.4420524, 1.4409602
7: -0.3616745, 0.4618227, -0.3651637, 0.4619819, -0.8236563, 0.8269864
8: -0.3729174, 0.4232192, -0.3766791, 0.4280382, -0.8009556, 0.7998983
9: -0.2493206, 0.2945979, -0.2533768, 0.3006897, -0.5500103, 0.5479748

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8991593, upper bound: 1.8901671
time: 2.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8991594, upper bound: 1.8905728
time: 3.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2197241, 0.9010219, -0.3918016, 1.2279290, -1.4476531, 1.2928234
1: -0.3088645, 0.3456420, -0.4751419, 0.4827785, -0.7916430, 0.8207840
2: -0.3928846, 0.4111367, -0.5660569, 0.5859142, -0.9787987, 0.9771937
3: -0.2772069, 0.2491153, -0.3856437, 0.4241588, -0.7013657, 0.6347591
4: -0.3066499, 0.3806731, -0.4956396, 0.5271402, -0.8337901, 0.8763126
5: -0.4323313, 0.4979123, -0.5913196, 0.6907331, -1.1230645, 1.0892318
6: -0.1188840, 1.2895670, -0.4771030, 1.4601940, -1.5790780, 1.7666700
7: -0.3478206, 0.4475762, -0.5397432, 0.6153690, -0.9631896, 0.9873194
8: -0.3624573, 0.4095375, -0.5397156, 0.6035459, -0.9660032, 0.9492531
9: -0.2377264, 0.2838125, -0.4020412, 0.4436079, -0.6813344, 0.6858537

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8894640
time: 2.11 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8897896
time: 2.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2360516, 0.9213381, -0.3536130, 1.1468372, -1.3828888, 1.2749511
1: -0.3221140, 0.3588542, -0.4354098, 0.4517244, -0.7738384, 0.7942641
2: -0.4051805, 0.4282180, -0.5241694, 0.5454540, -0.9506345, 0.9523875
3: -0.2861544, 0.2650071, -0.3605863, 0.3811630, -0.6673174, 0.6255934
4: -0.3222730, 0.3931868, -0.4508080, 0.4844708, -0.8067439, 0.8439949
5: -0.4428855, 0.5166113, -0.5507457, 0.6495175, -1.0924031, 1.0673571
6: -0.1440104, 1.2948549, -0.3902594, 1.4151530, -1.5591635, 1.6851143
7: -0.3651637, 0.4619819, -0.4937468, 0.5771936, -0.9423573, 0.9557287
8: -0.3766791, 0.4280382, -0.4922843, 0.5601331, -0.9368122, 0.9203224
9: -0.2533768, 0.3006897, -0.3607625, 0.4062229, -0.6595998, 0.6614522

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8897896, upper bound: 1.8900268
time: 2.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8897896, upper bound: 1.8905455
time: 2.48 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2172361, 0.9069000, -1.1571229, 2.1073875, -2.3246236, 2.0640230
1: -0.3086752, 0.3421522, -1.0190454, 0.9931424, -1.3018177, 1.3611975
2: -0.3938075, 0.4071273, -1.0939938, 1.3421152, -1.7359227, 1.5011210
3: -0.2772135, 0.2452471, -0.8575323, 1.0565397, -1.3337531, 1.1027794
4: -0.3061112, 0.3769857, -1.0527849, 1.4547693, -1.7608805, 1.4297707
5: -0.4345517, 0.4914089, -1.3438350, 1.4506361, -1.8851879, 1.8352439
6: -0.1216769, 1.2899801, -1.5473984, 1.8286030, -1.9502800, 2.8373785
7: -0.3441595, 0.4473241, -1.2758220, 1.1852087, -1.5293682, 1.7231462
8: -0.3585033, 0.4045241, -1.1452086, 1.4524922, -1.8109956, 1.5497327
9: -0.2335665, 0.2775798, -1.0130838, 1.1884294, -1.4219959, 1.2906637

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8246327
time: 1.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8248847
time: 1.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2336920, 0.9276679, -1.1064560, 2.0360427, -2.2697346, 2.0341239
1: -0.3219900, 0.3554728, -0.9828035, 0.9629667, -1.2849567, 1.3382763
2: -0.4061271, 0.4243887, -1.0578785, 1.2963203, -1.7024474, 1.4822673
3: -0.2861868, 0.2612684, -0.8345226, 1.0170850, -1.3032718, 1.0957910
4: -0.3218213, 0.3896247, -1.0146995, 1.3996853, -1.7215066, 1.4043242
5: -0.4451328, 0.5103453, -1.2906734, 1.4091601, -1.8542930, 1.8010187
6: -0.1471975, 1.2969497, -1.4684881, 1.7880827, -1.9352803, 2.7654378
7: -0.3616745, 0.4618227, -1.2301536, 1.1471224, -1.5087969, 1.6919763
8: -0.3729174, 0.4232192, -1.1096203, 1.4020764, -1.7749938, 1.5328395
9: -0.2493206, 0.2945979, -0.9756740, 1.1447871, -1.3941077, 1.2702719

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8898685, upper bound: 1.8313821
time: 2.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8898685, upper bound: 1.8319765
time: 1.96 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2197241, 0.9010219, -1.0967860, 2.0343475, -2.2540717, 1.9978080
1: -0.3088645, 0.3456420, -0.9765317, 0.9538596, -1.2627240, 1.3221737
2: -0.3928846, 0.4111367, -1.0522587, 1.2844235, -1.6773081, 1.4633955
3: -0.2772069, 0.2491153, -0.8183032, 1.0080730, -1.2852799, 1.0674186
4: -0.3066499, 0.3806731, -1.0088980, 1.3847561, -1.6914060, 1.3895711
5: -0.4323313, 0.4979123, -1.2836640, 1.3925086, -1.8248399, 1.7815763
6: -0.1188840, 1.2895670, -1.4616778, 1.7934395, -1.9123235, 2.7512448
7: -0.3478206, 0.4475762, -1.2195354, 1.1400954, -1.4879160, 1.6671115
8: -0.3624573, 0.4095375, -1.0968640, 1.3875718, -1.7500291, 1.5064015
9: -0.2377264, 0.2838125, -0.9673659, 1.1319284, -1.3696549, 1.2511784

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8245247
time: 1.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8248552
time: 1.99 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2360516, 0.9213381, -1.0426068, 1.9590895, -2.1951411, 1.9639449
1: -0.3221140, 0.3588542, -0.9377369, 0.9214756, -1.2435896, 1.2965912
2: -0.4051805, 0.4282180, -1.0136068, 1.2353890, -1.6405694, 1.4418248
3: -0.2861544, 0.2650071, -0.7930951, 0.9659141, -1.2520685, 1.0581021
4: -0.3222730, 0.3931868, -0.9682225, 1.3258679, -1.6481409, 1.3614093
5: -0.4428855, 0.5166113, -1.2270261, 1.3477001, -1.7905856, 1.7436373
6: -0.1440104, 1.2948549, -1.3777790, 1.7510053, -1.8950157, 2.6726339
7: -0.3651637, 0.4619819, -1.1707075, 1.0992603, -1.4644240, 1.6326894
8: -0.3766791, 0.4280382, -1.0584760, 1.3334701, -1.7101492, 1.4865141
9: -0.2533768, 0.3006897, -0.9275097, 1.0852115, -1.3385884, 1.2281994

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8831035, upper bound: 1.8312210
time: 1.76 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8831035, upper bound: 1.8319511
time: 2.22 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.1571229, 2.1073875, -0.2172361, 0.9069000, -2.0640230, 2.3246236
1: -1.0190454, 0.9931424, -0.3086752, 0.3421522, -1.3611975, 1.3018177
2: -1.0939938, 1.3421152, -0.3938075, 0.4071273, -1.5011210, 1.7359227
3: -0.8575323, 1.0565397, -0.2772135, 0.2452471, -1.1027794, 1.3337531
4: -1.0527849, 1.4547693, -0.3061112, 0.3769857, -1.4297707, 1.7608805
5: -1.3438350, 1.4506361, -0.4345517, 0.4914089, -1.8352439, 1.8851879
6: -1.5473984, 1.8286030, -0.1216769, 1.2899801, -2.8373785, 1.9502800
7: -1.2758220, 1.1852087, -0.3441595, 0.4473241, -1.7231462, 1.5293682
8: -1.1452086, 1.4524922, -0.3585033, 0.4045241, -1.5497327, 1.8109956
9: -1.0130838, 1.1884294, -0.2335665, 0.2775798, -1.2906637, 1.4219959

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8248493, upper bound: 1.8774045
time: 1.85 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8248493, upper bound: 1.8774045
time: 1.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.0967860, 2.0343475, -0.2197241, 0.9010219, -1.9978080, 2.2540717
1: -0.9765317, 0.9538596, -0.3088645, 0.3456420, -1.3221737, 1.2627240
2: -1.0522587, 1.2844235, -0.3928846, 0.4111367, -1.4633955, 1.6773081
3: -0.8183032, 1.0080730, -0.2772069, 0.2491153, -1.0674186, 1.2852799
4: -1.0088980, 1.3847561, -0.3066499, 0.3806731, -1.3895711, 1.6914060
5: -1.2836640, 1.3925086, -0.4323313, 0.4979123, -1.7815763, 1.8248399
6: -1.4616778, 1.7934395, -0.1188840, 1.2895670, -2.7512448, 1.9123235
7: -1.2195354, 1.1400954, -0.3478206, 0.4475762, -1.6671115, 1.4879160
8: -1.0968640, 1.3875718, -0.3624573, 0.4095375, -1.5064015, 1.7500291
9: -0.9673659, 1.1319284, -0.2377264, 0.2838125, -1.2511784, 1.3696549

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8248551, upper bound: 1.8774045
time: 2.02 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8248551, upper bound: 1.8774045
time: 2.17 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.1064560, 2.0360427, -0.2336920, 0.9276679, -2.0341239, 2.2697346
1: -0.9828035, 0.9629667, -0.3219900, 0.3554728, -1.3382763, 1.2849567
2: -1.0578785, 1.2963203, -0.4061271, 0.4243887, -1.4822673, 1.7024474
3: -0.8345226, 1.0170850, -0.2861868, 0.2612684, -1.0957910, 1.3032718
4: -1.0146995, 1.3996853, -0.3218213, 0.3896247, -1.4043242, 1.7215066
5: -1.2906734, 1.4091601, -0.4451328, 0.5103453, -1.8010187, 1.8542930
6: -1.4684881, 1.7880827, -0.1471975, 1.2969497, -2.7654378, 1.9352803
7: -1.2301536, 1.1471224, -0.3616745, 0.4618227, -1.6919763, 1.5087969
8: -1.1096203, 1.4020764, -0.3729174, 0.4232192, -1.5328395, 1.7749938
9: -0.9756740, 1.1447871, -0.2493206, 0.2945979, -1.2702719, 1.3941077

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8319449, upper bound: 1.8843870
time: 2.03 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8319449, upper bound: 1.8843870
time: 2.07 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.0426068, 1.9590895, -0.2360516, 0.9213381, -1.9639449, 2.1951411
1: -0.9377369, 0.9214756, -0.3221140, 0.3588542, -1.2965912, 1.2435896
2: -1.0136068, 1.2353890, -0.4051805, 0.4282180, -1.4418248, 1.6405694
3: -0.7930951, 0.9659141, -0.2861544, 0.2650071, -1.0581021, 1.2520685
4: -0.9682225, 1.3258679, -0.3222730, 0.3931868, -1.3614093, 1.6481409
5: -1.2270261, 1.3477001, -0.4428855, 0.5166113, -1.7436373, 1.7905856
6: -1.3777790, 1.7510053, -0.1440104, 1.2948549, -2.6726339, 1.8950157
7: -1.1707075, 1.0992603, -0.3651637, 0.4619819, -1.6326894, 1.4644240
8: -1.0584760, 1.3334701, -0.3766791, 0.4280382, -1.4865141, 1.7101492
9: -0.9275097, 1.0852115, -0.2533768, 0.3006897, -1.2281994, 1.3385884

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8319511, upper bound: 1.8843870
time: 2.65 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8319511, upper bound: 1.8843870
time: 2.04 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.1571229, 2.1073875, -0.8064981, 1.6604885, -2.8176112, 2.9138856
1: -1.0190454, 0.9931424, -0.7700347, 0.7722626, -1.7913079, 1.7631772
2: -1.0939938, 1.3421152, -0.8474082, 1.0151366, -2.1091304, 2.1895232
3: -0.8575323, 1.0565397, -0.6453625, 0.7806425, -1.6381748, 1.7019022
4: -1.0527849, 1.4547693, -0.7952353, 1.0636635, -2.1164484, 2.2500045
5: -1.3438350, 1.4506361, -0.9882920, 1.1254932, -2.4693282, 2.4389281
6: -1.5473984, 1.8286030, -1.0324848, 1.5929409, -3.1403394, 2.8610878
7: -1.2758220, 1.1852087, -0.9558080, 0.9207498, -2.1965718, 2.1410167
8: -1.1452086, 1.4524922, -0.8718588, 1.0849626, -2.2301712, 2.3243511
9: -1.0130838, 1.1884294, -0.7570274, 0.8727985, -1.8858824, 1.9454567

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8241022, upper bound: 1.8303278
time: 1.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8241022, upper bound: 1.8303279
time: 2.27 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.1064560, 2.0360427, -1.1571229, 2.1073875, -3.2138436, 3.1931655
1: -0.9828035, 0.9629667, -1.0190454, 0.9931424, -1.9759459, 1.9820120
2: -1.0578785, 1.2963203, -1.0939938, 1.3421152, -2.3999937, 2.3903141
3: -0.8345226, 1.0170850, -0.8575323, 1.0565397, -1.8910623, 1.8746173
4: -1.0146995, 1.3996853, -1.0527849, 1.4547693, -2.4694686, 2.4524703
5: -1.2906734, 1.4091601, -1.3438350, 1.4506361, -2.7413096, 2.7529950
6: -1.4684881, 1.7880827, -1.5473984, 1.8286030, -3.2970910, 3.3354812
7: -1.2301536, 1.1471224, -1.2758220, 1.1852087, -2.4153624, 2.4229445
8: -1.1096203, 1.4020764, -1.1452086, 1.4524922, -2.5621126, 2.5472851
9: -0.9756740, 1.1447871, -1.0130838, 1.1884294, -2.1641033, 2.1578708

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8459148, upper bound: 1.8241928
time: 2.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8303279, upper bound: 1.8241023
time: 1.93 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.1064560, 2.0360427, -1.1064560, 2.0360427, -3.1424987, 3.1424987
1: -0.9828035, 0.9629667, -0.9828035, 0.9629667, -1.9457703, 1.9457703
2: -1.0578785, 1.2963203, -1.0578785, 1.2963203, -2.3541989, 2.3541989
3: -0.8345226, 1.0170850, -0.8345226, 1.0170850, -1.8516076, 1.8516076
4: -1.0146995, 1.3996853, -1.0146995, 1.3996853, -2.4143848, 2.4143848
5: -1.2906734, 1.4091601, -1.2906734, 1.4091601, -2.6998334, 2.6998334
6: -1.4684881, 1.7880827, -1.4684881, 1.7880827, -3.2565708, 3.2565708
7: -1.2301536, 1.1471224, -1.2301536, 1.1471224, -2.3772759, 2.3772759
8: -1.1096203, 1.4020764, -1.1096203, 1.4020764, -2.5116968, 2.5116968
9: -0.9756740, 1.1447871, -0.9756740, 1.1447871, -2.1204610, 2.1204610

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8459148, upper bound: 1.8313277
time: 1.91 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8303279, upper bound: 1.8312450
time: 2.03 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.15 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8991593, upper bound: 1.8901671
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8991594, upper bound: 1.8905728
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8991593, upper bound: 1.8901671
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8991594, upper bound: 1.8905728
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8894640
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8897896
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8897896, upper bound: 1.8900268
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8897896, upper bound: 1.8905455
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8246327
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8248847
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8898685, upper bound: 1.8313821
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8898685, upper bound: 1.8319765
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8245247
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8248552
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8831035, upper bound: 1.8312210
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8831035, upper bound: 1.8319511
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8248493, upper bound: 1.8774045
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8248493, upper bound: 1.8774045
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8248551, upper bound: 1.8774045
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8248551, upper bound: 1.8774045
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8319449, upper bound: 1.8843870
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8319449, upper bound: 1.8843870
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8319511, upper bound: 1.8843870
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8319511, upper bound: 1.8843870
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8241022, upper bound: 1.8303278
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8241022, upper bound: 1.8303279
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8459148, upper bound: 1.8241928
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8303279, upper bound: 1.8241023
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8459148, upper bound: 1.8313277
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.15
Output dim: 6, lower bound: -1.8303279, upper bound: 1.8312450

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.2172361, 0.9069000, -1.1451691, 1.1680306
1: -0.3297816, 0.3564569, -0.3086752, 0.3421522, -0.6719338, 0.6651321
2: -0.4150444, 0.4270655, -0.3938075, 0.4071273, -0.8221717, 0.8208730
3: -0.2916944, 0.2636827, -0.2772135, 0.2452471, -0.5369415, 0.5408962
4: -0.3300874, 0.3899961, -0.3061112, 0.3769857, -0.7070731, 0.6961072
5: -0.4548751, 0.5089113, -0.4345517, 0.4914089, -0.9462839, 0.9434629
6: -0.1680785, 1.3109281, -0.1216769, 1.2899801, -1.4580586, 1.4326050
7: -0.3655823, 0.4693472, -0.3441595, 0.4473241, -0.8129064, 0.8135067
8: -0.3765907, 0.4246833, -0.3585033, 0.4045241, -0.7811148, 0.7831866
9: -0.2510976, 0.2932430, -0.2335665, 0.2775798, -0.5286774, 0.5268095

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8992347, upper bound: 1.8992347
time: 2.30 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8992347, upper bound: 1.8992347
time: 2.66 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.2336920, 0.9276679, -1.1327922, 1.1251924
1: -0.2987980, 0.3325009, -0.3219900, 0.3554728, -0.6542708, 0.6544909
2: -0.3846641, 0.3945331, -0.4061271, 0.4243887, -0.8090528, 0.8006602
3: -0.2706087, 0.2334781, -0.2861868, 0.2612684, -0.5318771, 0.5196649
4: -0.2944325, 0.3679506, -0.3218213, 0.3896247, -0.6840572, 0.6897718
5: -0.4267558, 0.4781152, -0.4451328, 0.5103453, -0.9371011, 0.9232481
6: -0.1026776, 1.2858521, -0.1471975, 1.2969497, -1.3996274, 1.4330497
7: -0.3313026, 0.4365475, -0.3616745, 0.4618227, -0.7931253, 0.7982220
8: -0.3483621, 0.3910702, -0.3729174, 0.4232192, -0.7715813, 0.7639877
9: -0.2218433, 0.2653263, -0.2493206, 0.2945979, -0.5164412, 0.5146468

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8992347, upper bound: 1.8995124
time: 2.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8992347, upper bound: 1.8995124
time: 2.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.2197241, 0.9010219, -1.1392910, 1.1705186
1: -0.3297816, 0.3564569, -0.3088645, 0.3456420, -0.6754236, 0.6653214
2: -0.4150444, 0.4270655, -0.3928846, 0.4111367, -0.8261811, 0.8199500
3: -0.2916944, 0.2636827, -0.2772069, 0.2491153, -0.5408098, 0.5408896
4: -0.3300874, 0.3899961, -0.3066499, 0.3806731, -0.7107604, 0.6966459
5: -0.4548751, 0.5089113, -0.4323313, 0.4979123, -0.9527873, 0.9412426
6: -0.1680785, 1.3109281, -0.1188840, 1.2895670, -1.4576455, 1.4298121
7: -0.3655823, 0.4693472, -0.3478206, 0.4475762, -0.8131585, 0.8171678
8: -0.3765907, 0.4246833, -0.3624573, 0.4095375, -0.7861282, 0.7871406
9: -0.2510976, 0.2932430, -0.2377264, 0.2838125, -0.5349101, 0.5309694

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8984257, upper bound: 1.8895620
time: 3.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8984257, upper bound: 1.8901671
time: 2.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.2360516, 0.9213381, -1.1264626, 1.1275519
1: -0.2987980, 0.3325009, -0.3221140, 0.3588542, -0.6576523, 0.6546149
2: -0.3846641, 0.3945331, -0.4051805, 0.4282180, -0.8128821, 0.7997136
3: -0.2706087, 0.2334781, -0.2861544, 0.2650071, -0.5356159, 0.5196325
4: -0.2944325, 0.3679506, -0.3222730, 0.3931868, -0.6876193, 0.6902236
5: -0.4267558, 0.4781152, -0.4428855, 0.5166113, -0.9433671, 0.9210007
6: -0.1026776, 1.2858521, -0.1440104, 1.2948549, -1.3975325, 1.4298625
7: -0.3313026, 0.4365475, -0.3651637, 0.4619819, -0.7932844, 0.8017112
8: -0.3483621, 0.3910702, -0.3766791, 0.4280382, -0.7764002, 0.7677493
9: -0.2218433, 0.2653263, -0.2533768, 0.3006897, -0.5225329, 0.5187031

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8984257, upper bound: 1.8898142
time: 1.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8984257, upper bound: 1.8905729
time: 2.16 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -0.3918016, 1.2279290, -1.4844842, 1.3662418
1: -0.3448424, 0.3715456, -0.4751419, 0.4827785, -0.8276210, 0.8466876
2: -0.4289511, 0.4466715, -0.5660569, 0.5859142, -1.0148653, 1.0127283
3: -0.3015637, 0.2813208, -0.3856437, 0.4241588, -0.7257226, 0.6669645
4: -0.3472072, 0.4048828, -0.4956396, 0.5271402, -0.8743473, 0.9005224
5: -0.4669372, 0.5324135, -0.5913196, 0.6907331, -1.1576703, 1.1237330
6: -0.1975362, 1.3229021, -0.4771030, 1.4601940, -1.6577301, 1.8000051
7: -0.3851250, 0.4853222, -0.5397432, 0.6153690, -1.0004940, 1.0250654
8: -0.3923242, 0.4465339, -0.5397156, 0.6035459, -0.9958701, 0.9862494
9: -0.2674738, 0.3125915, -0.4020412, 0.4436079, -0.7110817, 0.7146327

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8894640
time: 2.09 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8894640
time: 2.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2067769, 0.8844675, -0.3918016, 1.2279290, -1.4347059, 1.2762691
1: -0.2982824, 0.3353040, -0.4751419, 0.4827785, -0.7810610, 0.8104459
2: -0.3830329, 0.3976636, -0.5660569, 0.5859142, -0.9689471, 0.9637206
3: -0.2701186, 0.2365419, -0.3856437, 0.4241588, -0.6942774, 0.6221856
4: -0.2941514, 0.3709590, -0.4956396, 0.5271402, -0.8212916, 0.8665986
5: -0.4239522, 0.4835648, -0.5913196, 0.6907331, -1.1146853, 1.0748844
6: -0.0985513, 1.2864097, -0.4771030, 1.4601940, -1.5587454, 1.7635127
7: -0.3340706, 0.4360552, -0.5397432, 0.6153690, -0.9494396, 0.9757985
8: -0.3514939, 0.3950947, -0.5397156, 0.6035459, -0.9550397, 0.9348103
9: -0.2252686, 0.2706831, -0.4020412, 0.4436079, -0.6688765, 0.6727242

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8894640
time: 1.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8897896
time: 2.56 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -0.3536130, 1.1468372, -1.4033923, 1.3280532
1: -0.3448424, 0.3715456, -0.4354098, 0.4517244, -0.7965668, 0.8069554
2: -0.4289511, 0.4466715, -0.5241694, 0.5454540, -0.9744052, 0.9708409
3: -0.3015637, 0.2813208, -0.3605863, 0.3811630, -0.6827267, 0.6419071
4: -0.3472072, 0.4048828, -0.4508080, 0.4844708, -0.8316780, 0.8556908
5: -0.4669372, 0.5324135, -0.5507457, 0.6495175, -1.1164547, 1.0831592
6: -0.1975362, 1.3229021, -0.3902594, 1.4151530, -1.6126893, 1.7131615
7: -0.3851250, 0.4853222, -0.4937468, 0.5771936, -0.9623186, 0.9790689
8: -0.3923242, 0.4465339, -0.4922843, 0.5601331, -0.9524573, 0.9388181
9: -0.2674738, 0.3125915, -0.3607625, 0.4062229, -0.6736968, 0.6733540

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8900268
time: 2.37 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8900268
time: 2.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2067769, 0.8844675, -0.3536130, 1.1468372, -1.3536141, 1.2380805
1: -0.2982824, 0.3353040, -0.4354098, 0.4517244, -0.7500069, 0.7707138
2: -0.3830329, 0.3976636, -0.5241694, 0.5454540, -0.9284869, 0.9218330
3: -0.2701186, 0.2365419, -0.3605863, 0.3811630, -0.6512816, 0.5971282
4: -0.2941514, 0.3709590, -0.4508080, 0.4844708, -0.7786222, 0.8217671
5: -0.4239522, 0.4835648, -0.5507457, 0.6495175, -1.0734698, 1.0343106
6: -0.0985513, 1.2864097, -0.3902594, 1.4151530, -1.5137043, 1.6766691
7: -0.3340706, 0.4360552, -0.4937468, 0.5771936, -0.9112642, 0.9298020
8: -0.3514939, 0.3950947, -0.4922843, 0.5601331, -0.9116269, 0.8873789
9: -0.2252686, 0.2706831, -0.3607625, 0.4062229, -0.6314915, 0.6314456

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8905455
time: 2.13 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8905455
time: 2.00 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -1.1571229, 2.1073875, -2.3456566, 2.1079173
1: -0.3297816, 0.3564569, -1.0190454, 0.9931424, -1.3229240, 1.3755023
2: -0.4150444, 0.4270655, -1.0939938, 1.3421152, -1.7571595, 1.5210593
3: -0.2916944, 0.2636827, -0.8575323, 1.0565397, -1.3482341, 1.1212151
4: -0.3300874, 0.3899961, -1.0527849, 1.4547693, -1.7848566, 1.4427810
5: -0.4548751, 0.5089113, -1.3438350, 1.4506361, -1.9055111, 1.8527462
6: -0.1680785, 1.3109281, -1.5473984, 1.8286030, -1.9966816, 2.8583264
7: -0.3655823, 0.4693472, -1.2758220, 1.1852087, -1.5507910, 1.7451692
8: -0.3765907, 0.4246833, -1.1452086, 1.4524922, -1.8290830, 1.5698919
9: -0.2510976, 0.2932430, -1.0130838, 1.1884294, -1.4395269, 1.3063269

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 122

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8246327
time: 2.24 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8246327
time: 2.11 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -1.1571229, 2.1073875, -2.3125119, 2.0486231
1: -0.2987980, 0.3325009, -1.0190454, 0.9931424, -1.2919405, 1.3515463
2: -0.3846641, 0.3945331, -1.0939938, 1.3421152, -1.7267792, 1.4885269
3: -0.2706087, 0.2334781, -0.8575323, 1.0565397, -1.3271484, 1.0910105
4: -0.2944325, 0.3679506, -1.0527849, 1.4547693, -1.7492018, 1.4207355
5: -0.4267558, 0.4781152, -1.3438350, 1.4506361, -1.8773919, 1.8219502
6: -0.1026776, 1.2858521, -1.5473984, 1.8286030, -1.9312806, 2.8332505
7: -0.3313026, 0.4365475, -1.2758220, 1.1852087, -1.5165112, 1.7123696
8: -0.3483621, 0.3910702, -1.1452086, 1.4524922, -1.8008543, 1.5362788
9: -0.2218433, 0.2653263, -1.0130838, 1.1884294, -1.4102726, 1.2784101

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 122

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8248847
time: 2.22 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8248847
time: 2.27 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -1.1064560, 2.0360427, -2.2743118, 2.0572505
1: -0.3297816, 0.3564569, -0.9828035, 0.9629667, -1.2927483, 1.3392603
2: -0.4150444, 0.4270655, -1.0578785, 1.2963203, -1.7113647, 1.4849440
3: -0.2916944, 0.2636827, -0.8345226, 1.0170850, -1.3087794, 1.0982053
4: -0.3300874, 0.3899961, -1.0146995, 1.3996853, -1.7297726, 1.4046955
5: -0.4548751, 0.5089113, -1.2906734, 1.4091601, -1.8640351, 1.7995846
6: -0.1680785, 1.3109281, -1.4684881, 1.7880827, -1.9561613, 2.7794161
7: -0.3655823, 0.4693472, -1.2301536, 1.1471224, -1.5127047, 1.6995008
8: -0.3765907, 0.4246833, -1.1096203, 1.4020764, -1.7786671, 1.5343037
9: -0.2510976, 0.2932430, -0.9756740, 1.1447871, -1.3958846, 1.2689170

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 122

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8313821
time: 1.93 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8313821
time: 2.24 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -1.1064560, 2.0360427, -2.2411671, 1.9979564
1: -0.2987980, 0.3325009, -0.9828035, 0.9629667, -1.2617648, 1.3153043
2: -0.3846641, 0.3945331, -1.0578785, 1.2963203, -1.6809844, 1.4524117
3: -0.2706087, 0.2334781, -0.8345226, 1.0170850, -1.2876937, 1.0680007
4: -0.2944325, 0.3679506, -1.0146995, 1.3996853, -1.6941178, 1.3826500
5: -0.4267558, 0.4781152, -1.2906734, 1.4091601, -1.8359159, 1.7687886
6: -0.1026776, 1.2858521, -1.4684881, 1.7880827, -1.8907604, 2.7543402
7: -0.3313026, 0.4365475, -1.2301536, 1.1471224, -1.4784249, 1.6667011
8: -0.3483621, 0.3910702, -1.1096203, 1.4020764, -1.7504385, 1.5006906
9: -0.2218433, 0.2653263, -0.9756740, 1.1447871, -1.3666303, 1.2410002

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 122

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8319764
time: 2.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8319764
time: 1.99 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -1.0967860, 2.0343475, -2.2909026, 2.0712261
1: -0.3448424, 0.3715456, -0.9765317, 0.9538596, -1.2987020, 1.3480773
2: -0.4289511, 0.4466715, -1.0522587, 1.2844235, -1.7133746, 1.4989302
3: -0.3015637, 0.2813208, -0.8183032, 1.0080730, -1.3096367, 1.0996240
4: -0.3472072, 0.4048828, -1.0088980, 1.3847561, -1.7319633, 1.4137808
5: -0.4669372, 0.5324135, -1.2836640, 1.3925086, -1.8594458, 1.8160775
6: -0.1975362, 1.3229021, -1.4616778, 1.7934395, -1.9909756, 2.7845798
7: -0.3851250, 0.4853222, -1.2195354, 1.1400954, -1.5252204, 1.7048576
8: -0.3923242, 0.4465339, -1.0968640, 1.3875718, -1.7798960, 1.5433979
9: -0.2674738, 0.3125915, -0.9673659, 1.1319284, -1.3994023, 1.2799574

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 122

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8245247
time: 2.16 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8245247
time: 2.29 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2067769, 0.8844675, -1.0967860, 2.0343475, -2.2411244, 1.9812535
1: -0.2982824, 0.3353040, -0.9765317, 0.9538596, -1.2521420, 1.3118356
2: -0.3830329, 0.3976636, -1.0522587, 1.2844235, -1.6674564, 1.4499223
3: -0.2701186, 0.2365419, -0.8183032, 1.0080730, -1.2781916, 1.0548451
4: -0.2941514, 0.3709590, -1.0088980, 1.3847561, -1.6789074, 1.3798571
5: -0.4239522, 0.4835648, -1.2836640, 1.3925086, -1.8164608, 1.7672288
6: -0.0985513, 1.2864097, -1.4616778, 1.7934395, -1.8919909, 2.7480874
7: -0.3340706, 0.4360552, -1.2195354, 1.1400954, -1.4741659, 1.6555905
8: -0.3514939, 0.3950947, -1.0968640, 1.3875718, -1.7390656, 1.4919586
9: -0.2252686, 0.2706831, -0.9673659, 1.1319284, -1.3571970, 1.2380489

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 122

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8248552
time: 2.03 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8248551
time: 2.33 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -1.0426068, 1.9590895, -2.2156446, 2.0170469
1: -0.3448424, 0.3715456, -0.9377369, 0.9214756, -1.2663181, 1.3092825
2: -0.4289511, 0.4466715, -1.0136068, 1.2353890, -1.6643401, 1.4602783
3: -0.3015637, 0.2813208, -0.7930951, 0.9659141, -1.2674779, 1.0744159
4: -0.3472072, 0.4048828, -0.9682225, 1.3258679, -1.6730751, 1.3731053
5: -0.4669372, 0.5324135, -1.2270261, 1.3477001, -1.8146373, 1.7594396
6: -0.1975362, 1.3229021, -1.3777790, 1.7510053, -1.9485414, 2.7006812
7: -0.3851250, 0.4853222, -1.1707075, 1.0992603, -1.4843853, 1.6560297
8: -0.3923242, 0.4465339, -1.0584760, 1.3334701, -1.7257943, 1.5050099
9: -0.2674738, 0.3125915, -0.9275097, 1.0852115, -1.3526853, 1.2401012

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 122

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8312210
time: 1.98 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8312210
time: 2.21 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2067769, 0.8844675, -1.0426068, 1.9590895, -2.1658664, 1.9270743
1: -0.2982824, 0.3353040, -0.9377369, 0.9214756, -1.2197580, 1.2730409
2: -0.3830329, 0.3976636, -1.0136068, 1.2353890, -1.6184219, 1.4112704
3: -0.2701186, 0.2365419, -0.7930951, 0.9659141, -1.2360327, 1.0296369
4: -0.2941514, 0.3709590, -0.9682225, 1.3258679, -1.6200192, 1.3391815
5: -0.4239522, 0.4835648, -1.2270261, 1.3477001, -1.7716523, 1.7105908
6: -0.0985513, 1.2864097, -1.3777790, 1.7510053, -1.8495567, 2.6641889
7: -0.3340706, 0.4360552, -1.1707075, 1.0992603, -1.4333310, 1.6067626
8: -0.3514939, 0.3950947, -1.0584760, 1.3334701, -1.6849639, 1.4535706
9: -0.2252686, 0.2706831, -0.9275097, 1.0852115, -1.3104801, 1.1981928

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 122

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8319511
time: 1.61 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8319511
time: 1.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.8261355, 1.6957741, -0.2172361, 0.9069000, -1.7330356, 1.9130101
1: -0.7845350, 0.7813374, -0.3086752, 0.3421522, -1.1266872, 1.0900126
2: -0.8624935, 1.0301049, -0.3938075, 0.4071273, -1.2696208, 1.4239124
3: -0.6554159, 0.7940063, -0.2772135, 0.2452471, -0.9006630, 1.0712199
4: -0.8107981, 1.0799718, -0.3061112, 0.3769857, -1.1877838, 1.3860829
5: -1.0105178, 1.1361258, -0.4345517, 0.4914089, -1.5019267, 1.5706775
6: -1.0676947, 1.6187458, -0.1216769, 1.2899801, -2.3576746, 1.7404227
7: -0.9711751, 0.9359220, -0.3441595, 0.4473241, -1.4184992, 1.2800815
8: -0.8822247, 1.1008432, -0.3585033, 0.4045241, -1.2867488, 1.4593465
9: -0.7688699, 0.8852291, -0.2335665, 0.2775798, -1.0464498, 1.1187956

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8246327, upper bound: 1.8822187
time: 2.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8246327, upper bound: 1.8822187
time: 2.28 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.8663305, 1.7437507, -0.2172361, 0.9069000, -1.7732306, 1.9609867
1: -0.8130708, 0.8073345, -0.3086752, 0.3421522, -1.1552230, 1.1160097
2: -0.8903426, 1.0682744, -0.3938075, 0.4071273, -1.2974699, 1.4620819
3: -0.6797470, 0.8258173, -0.2772135, 0.2452471, -0.9249941, 1.1030309
4: -0.8397647, 1.1246009, -0.3061112, 0.3769857, -1.2167504, 1.4307120
5: -1.0506577, 1.1777372, -0.4345517, 0.4914089, -1.5420666, 1.6122890
6: -1.1253142, 1.6426497, -0.1216769, 1.2899801, -2.4152942, 1.7643266
7: -1.0080607, 0.9664483, -0.3441595, 0.4473241, -1.4553847, 1.3106079
8: -0.9167264, 1.1443964, -0.3585033, 0.4045241, -1.3212504, 1.5028996
9: -0.7970768, 0.9224179, -0.2335665, 0.2775798, -1.0746567, 1.1559844

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8246327, upper bound: 1.8822187
time: 2.44 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8246327, upper bound: 1.8822187
time: 2.32 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.8261355, 1.6957741, -0.2197241, 0.9010219, -1.7271574, 1.9154981
1: -0.7845350, 0.7813374, -0.3088645, 0.3456420, -1.1301770, 1.0902019
2: -0.8624935, 1.0301049, -0.3928846, 0.4111367, -1.2736301, 1.4229894
3: -0.6554159, 0.7940063, -0.2772069, 0.2491153, -0.9045312, 1.0712132
4: -0.8107981, 1.0799718, -0.3066499, 0.3806731, -1.1914711, 1.3866217
5: -1.0105178, 1.1361258, -0.4323313, 0.4979123, -1.5084301, 1.5684571
6: -1.0676947, 1.6187458, -0.1188840, 1.2895670, -2.3572617, 1.7376298
7: -0.9711751, 0.9359220, -0.3478206, 0.4475762, -1.4187512, 1.2837427
8: -0.8822247, 1.1008432, -0.3624573, 0.4095375, -1.2917621, 1.4633005
9: -0.7688699, 0.8852291, -0.2377264, 0.2838125, -1.0526824, 1.1229556

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8245247, upper bound: 1.8773133
time: 1.78 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8245247, upper bound: 1.8774045
time: 2.17 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.8663305, 1.7437507, -0.2197241, 0.9010219, -1.7673523, 1.9634748
1: -0.8130708, 0.8073345, -0.3088645, 0.3456420, -1.1587129, 1.1161990
2: -0.8903426, 1.0682744, -0.3928846, 0.4111367, -1.3014793, 1.4611590
3: -0.6797470, 0.8258173, -0.2772069, 0.2491153, -0.9288624, 1.1030242
4: -0.8397647, 1.1246009, -0.3066499, 0.3806731, -1.2204378, 1.4312508
5: -1.0506577, 1.1777372, -0.4323313, 0.4979123, -1.5485700, 1.6100686
6: -1.1253142, 1.6426497, -0.1188840, 1.2895670, -2.4148812, 1.7615336
7: -1.0080607, 0.9664483, -0.3478206, 0.4475762, -1.4556369, 1.3142689
8: -0.9167264, 1.1443964, -0.3624573, 0.4095375, -1.3262639, 1.5068537
9: -0.7970768, 0.9224179, -0.2377264, 0.2838125, -1.0808893, 1.1601443

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8245247, upper bound: 1.8773133
time: 2.26 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8245247, upper bound: 1.8774045
time: 2.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.7844378, 1.6371753, -0.2336920, 0.9276679, -1.7121056, 1.8708674
1: -0.7545584, 0.7572792, -0.3219900, 0.3554728, -1.1100311, 1.0792692
2: -0.8325227, 0.9933141, -0.4061271, 0.4243887, -1.2569115, 1.3994411
3: -0.6313741, 0.7623650, -0.2861868, 0.2612684, -0.8926425, 1.0485518
4: -0.7793881, 1.0365195, -0.3218213, 0.3896247, -1.1690128, 1.3583407
5: -0.9667646, 1.1034745, -0.4451328, 0.5103453, -1.4771099, 1.5486073
6: -1.0031563, 1.5844967, -0.1471975, 1.2969497, -2.3001060, 1.7316942
7: -0.9345270, 0.9043884, -0.3616745, 0.4618227, -1.3963498, 1.2660629
8: -0.8537888, 1.0604813, -0.3729174, 0.4232192, -1.2770081, 1.4333987
9: -0.7392236, 0.8510032, -0.2493206, 0.2945979, -1.0338216, 1.1003238

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8313821, upper bound: 1.8898685
time: 2.29 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8313821, upper bound: 1.8900152
time: 2.48 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7919115, 1.6434530, -0.2336920, 0.9276679, -1.7195793, 1.8771451
1: -0.7598349, 0.7628895, -0.3219900, 0.3554728, -1.1153077, 1.0848795
2: -0.8372648, 1.0011079, -0.4061271, 0.4243887, -1.2616535, 1.4072350
3: -0.6400046, 0.7686682, -0.2861868, 0.2612684, -0.9012730, 1.0548549
4: -0.7842556, 1.0456074, -0.3218213, 0.3896247, -1.1738803, 1.3674288
5: -0.9735982, 1.1139143, -0.4451328, 0.5103453, -1.4839435, 1.5590471
6: -1.0126145, 1.5860882, -0.1471975, 1.2969497, -2.3095641, 1.7332857
7: -0.9417365, 0.9100840, -0.3616745, 0.4618227, -1.4035592, 1.2717586
8: -0.8621641, 1.0699773, -0.3729174, 0.4232192, -1.2853833, 1.4428947
9: -0.7442783, 0.8592910, -0.2493206, 0.2945979, -1.0388763, 1.1086116

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8313821, upper bound: 1.8898685
time: 2.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8313821, upper bound: 1.8900152
time: 2.30 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.7844378, 1.6371753, -0.2360516, 0.9213381, -1.7057760, 1.8732269
1: -0.7545584, 0.7572792, -0.3221140, 0.3588542, -1.1134126, 1.0793931
2: -0.8325227, 0.9933141, -0.4051805, 0.4282180, -1.2607408, 1.3984946
3: -0.6313741, 0.7623650, -0.2861544, 0.2650071, -0.8963813, 1.0485194
4: -0.7793881, 1.0365195, -0.3222730, 0.3931868, -1.1725749, 1.3587925
5: -0.9667646, 1.1034745, -0.4428855, 0.5166113, -1.4833758, 1.5463600
6: -1.0031563, 1.5844967, -0.1440104, 1.2948549, -2.2980113, 1.7285072
7: -0.9345270, 0.9043884, -0.3651637, 0.4619819, -1.3965089, 1.2695520
8: -0.8537888, 1.0604813, -0.3766791, 0.4280382, -1.2818270, 1.4371604
9: -0.7392236, 0.8510032, -0.2533768, 0.3006897, -1.0399133, 1.1043801

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8312210, upper bound: 1.8831035
time: 2.51 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8312210, upper bound: 1.8843870
time: 2.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7919115, 1.6434530, -0.2360516, 0.9213381, -1.7132497, 1.8795046
1: -0.7598349, 0.7628895, -0.3221140, 0.3588542, -1.1186891, 1.0850035
2: -0.8372648, 1.0011079, -0.4051805, 0.4282180, -1.2654828, 1.4062884
3: -0.6400046, 0.7686682, -0.2861544, 0.2650071, -0.9050118, 1.0548226
4: -0.7842556, 1.0456074, -0.3222730, 0.3931868, -1.1774423, 1.3678805
5: -0.9735982, 1.1139143, -0.4428855, 0.5166113, -1.4902095, 1.5567998
6: -1.0126145, 1.5860882, -0.1440104, 1.2948549, -2.3074694, 1.7300986
7: -0.9417365, 0.9100840, -0.3651637, 0.4619819, -1.4037184, 1.2752477
8: -0.8621641, 1.0699773, -0.3766791, 0.4280382, -1.2902023, 1.4466563
9: -0.7442783, 0.8592910, -0.2533768, 0.3006897, -1.0449680, 1.1126678

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8312210, upper bound: 1.8831035
time: 1.99 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8312210, upper bound: 1.8843870
time: 2.01 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.7844378, 1.6371753, -1.1571229, 2.1073875, -2.8918252, 2.7942982
1: -0.7545584, 0.7572792, -1.0190454, 0.9931424, -1.7477008, 1.7763245
2: -0.8325227, 0.9933141, -1.0939938, 1.3421152, -2.1746378, 2.0873079
3: -0.6313741, 0.7623650, -0.8575323, 1.0565397, -1.6879138, 1.6198974
4: -0.7793881, 1.0365195, -1.0527849, 1.4547693, -2.2341573, 2.0893044
5: -0.9667646, 1.1034745, -1.3438350, 1.4506361, -2.4174008, 2.4473095
6: -1.0031563, 1.5844967, -1.5473984, 1.8286030, -2.8317595, 3.1318951
7: -0.9345270, 0.9043884, -1.2758220, 1.1852087, -2.1197357, 2.1802104
8: -0.8537888, 1.0604813, -1.1452086, 1.4524922, -2.3062811, 2.2056899
9: -0.7392236, 0.8510032, -1.0130838, 1.1884294, -1.9276530, 1.8640871

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 122

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8303278, upper bound: 1.8241023
time: 1.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8303278, upper bound: 1.8241023
time: 1.85 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.7844378, 1.6371753, -1.1064560, 2.0360427, -2.8204803, 2.7436314
1: -0.7545584, 0.7572792, -0.9828035, 0.9629667, -1.7175251, 1.7400826
2: -0.8325227, 0.9933141, -1.0578785, 1.2963203, -2.1288431, 2.0511925
3: -0.6313741, 0.7623650, -0.8345226, 1.0170850, -1.6484591, 1.5968876
4: -0.7793881, 1.0365195, -1.0146995, 1.3996853, -2.1790733, 2.0512190
5: -0.9667646, 1.1034745, -1.2906734, 1.4091601, -2.3759246, 2.3941479
6: -1.0031563, 1.5844967, -1.4684881, 1.7880827, -2.7912390, 3.0529847
7: -0.9345270, 0.9043884, -1.2301536, 1.1471224, -2.0816493, 2.1345420
8: -0.8537888, 1.0604813, -1.1096203, 1.4020764, -2.2558651, 2.1701016
9: -0.7392236, 0.8510032, -0.9756740, 1.1447871, -1.8840107, 1.8266772

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 122

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8311155, upper bound: 1.8312450
time: 1.96 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8311155, upper bound: 1.8312450
time: 2.26 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.55 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8992347, upper bound: 1.8992347
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8992347, upper bound: 1.8992347
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8992347, upper bound: 1.8995124
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8992347, upper bound: 1.8995124
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8984257, upper bound: 1.8895620
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8984257, upper bound: 1.8901671
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8984257, upper bound: 1.8898142
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8984257, upper bound: 1.8905729
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8894640
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8894640
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8894640
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8897896
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8900268
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8900268
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8905455
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8905455
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8246327
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8246327
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8248847
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8248847
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8313821
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8313821
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8319764
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8319764
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8245247
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8245247
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8248552
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8248551
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8312210
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8312210
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8319511
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8319511
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8246327, upper bound: 1.8822187
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8246327, upper bound: 1.8822187
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8246327, upper bound: 1.8822187
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8246327, upper bound: 1.8822187
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8245247, upper bound: 1.8773133
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8245247, upper bound: 1.8774045
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8245247, upper bound: 1.8773133
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8245247, upper bound: 1.8774045
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8313821, upper bound: 1.8898685
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8313821, upper bound: 1.8900152
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8313821, upper bound: 1.8898685
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8313821, upper bound: 1.8900152
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8312210, upper bound: 1.8831035
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8312210, upper bound: 1.8843870
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8312210, upper bound: 1.8831035
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8312210, upper bound: 1.8843870
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8303278, upper bound: 1.8241023
IS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8303278, upper bound: 1.8241023
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8311155, upper bound: 1.8312450
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.55
Output dim: 6, lower bound: -1.8311155, upper bound: 1.8312450

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.2382691, 0.9507946, -1.1890637, 1.1890637
1: -0.3297816, 0.3564569, -0.3297816, 0.3564569, -0.6862385, 0.6862385
2: -0.4150444, 0.4270655, -0.4150444, 0.4270655, -0.8421099, 0.8421099
3: -0.2916944, 0.2636827, -0.2916944, 0.2636827, -0.5553771, 0.5553771
4: -0.3300874, 0.3899961, -0.3300874, 0.3899961, -0.7200834, 0.7200834
5: -0.4548751, 0.5089113, -0.4548751, 0.5089113, -0.9637863, 0.9637863
6: -0.1680785, 1.3109281, -0.1680785, 1.3109281, -1.4790066, 1.4790066
7: -0.3655823, 0.4693472, -0.3655823, 0.4693472, -0.8349295, 0.8349295
8: -0.3765907, 0.4246833, -0.3765907, 0.4246833, -0.8012741, 0.8012741
9: -0.2510976, 0.2932430, -0.2510976, 0.2932430, -0.5443406, 0.5443406

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8992347, upper bound: 1.8986000
time: 2.24 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8992347, upper bound: 1.8992347
time: 2.09 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.2051244, 0.8915004, -1.1297694, 1.1559190
1: -0.3297816, 0.3564569, -0.2987980, 0.3325009, -0.6622825, 0.6552550
2: -0.4150444, 0.4270655, -0.3846641, 0.3945331, -0.8095775, 0.8117296
3: -0.2916944, 0.2636827, -0.2706087, 0.2334781, -0.5251725, 0.5342914
4: -0.3300874, 0.3899961, -0.2944325, 0.3679506, -0.6980379, 0.6844286
5: -0.4548751, 0.5089113, -0.4267558, 0.4781152, -0.9329903, 0.9356670
6: -0.1680785, 1.3109281, -0.1026776, 1.2858521, -1.4539306, 1.4136057
7: -0.3655823, 0.4693472, -0.3313026, 0.4365475, -0.8021299, 0.8006498
8: -0.3765907, 0.4246833, -0.3483621, 0.3910702, -0.7676610, 0.7730454
9: -0.2510976, 0.2932430, -0.2218433, 0.2653263, -0.5164238, 0.5150863

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8992347, upper bound: 1.8986000
time: 2.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8992347, upper bound: 1.8992347
time: 2.24 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.2382691, 0.9507946, -1.1559190, 1.1297694
1: -0.2987980, 0.3325009, -0.3297816, 0.3564569, -0.6552550, 0.6622825
2: -0.3846641, 0.3945331, -0.4150444, 0.4270655, -0.8117296, 0.8095775
3: -0.2706087, 0.2334781, -0.2916944, 0.2636827, -0.5342914, 0.5251725
4: -0.2944325, 0.3679506, -0.3300874, 0.3899961, -0.6844286, 0.6980379
5: -0.4267558, 0.4781152, -0.4548751, 0.5089113, -0.9356670, 0.9329903
6: -0.1026776, 1.2858521, -0.1680785, 1.3109281, -1.4136057, 1.4539306
7: -0.3313026, 0.4365475, -0.3655823, 0.4693472, -0.8006498, 0.8021299
8: -0.3483621, 0.3910702, -0.3765907, 0.4246833, -0.7730454, 0.7676610
9: -0.2218433, 0.2653263, -0.2510976, 0.2932430, -0.5150863, 0.5164238

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8992347, upper bound: 1.8990102
time: 2.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8992347, upper bound: 1.8995124
time: 2.19 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.2051244, 0.8915004, -1.0966247, 1.0966247
1: -0.2987980, 0.3325009, -0.2987980, 0.3325009, -0.6312989, 0.6312989
2: -0.3846641, 0.3945331, -0.3846641, 0.3945331, -0.7791972, 0.7791972
3: -0.2706087, 0.2334781, -0.2706087, 0.2334781, -0.5040869, 0.5040869
4: -0.2944325, 0.3679506, -0.2944325, 0.3679506, -0.6623831, 0.6623831
5: -0.4267558, 0.4781152, -0.4267558, 0.4781152, -0.9048710, 0.9048710
6: -0.1026776, 1.2858521, -0.1026776, 1.2858521, -1.3885298, 1.3885298
7: -0.3313026, 0.4365475, -0.3313026, 0.4365475, -0.7678500, 0.7678500
8: -0.3483621, 0.3910702, -0.3483621, 0.3910702, -0.7394323, 0.7394323
9: -0.2218433, 0.2653263, -0.2218433, 0.2653263, -0.4871695, 0.4871695

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8992347, upper bound: 1.8990102
time: 2.13 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8992347, upper bound: 1.8995124
time: 2.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.2565551, 0.9744402, -1.2127093, 1.2073498
1: -0.3297816, 0.3564569, -0.3448424, 0.3715456, -0.7013272, 0.7012994
2: -0.4150444, 0.4270655, -0.4289511, 0.4466715, -0.8617158, 0.8560166
3: -0.2916944, 0.2636827, -0.3015637, 0.2813208, -0.5730152, 0.5652465
4: -0.3300874, 0.3899961, -0.3472072, 0.4048828, -0.7349701, 0.7372032
5: -0.4548751, 0.5089113, -0.4669372, 0.5324135, -0.9872885, 0.9758484
6: -0.1680785, 1.3109281, -0.1975362, 1.3229021, -1.4909806, 1.5084643
7: -0.3655823, 0.4693472, -0.3851250, 0.4853222, -0.8509045, 0.8544722
8: -0.3765907, 0.4246833, -0.3923242, 0.4465339, -0.8231246, 0.8170075
9: -0.2510976, 0.2932430, -0.2674738, 0.3125915, -0.5636891, 0.5607168

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8984203, upper bound: 1.8888604
time: 1.92 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8984203, upper bound: 1.8895620
time: 1.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.2067769, 0.8844675, -1.1227366, 1.1575714
1: -0.3297816, 0.3564569, -0.2982824, 0.3353040, -0.6650856, 0.6547394
2: -0.4150444, 0.4270655, -0.3830329, 0.3976636, -0.8127080, 0.8100984
3: -0.2916944, 0.2636827, -0.2701186, 0.2365419, -0.5282363, 0.5338013
4: -0.3300874, 0.3899961, -0.2941514, 0.3709590, -0.7010463, 0.6841474
5: -0.4548751, 0.5089113, -0.4239522, 0.4835648, -0.9384398, 0.9328635
6: -0.1680785, 1.3109281, -0.0985513, 1.2864097, -1.4544883, 1.4094794
7: -0.3655823, 0.4693472, -0.3340706, 0.4360552, -0.8016376, 0.8034178
8: -0.3765907, 0.4246833, -0.3514939, 0.3950947, -0.7716854, 0.7761772
9: -0.2510976, 0.2932430, -0.2252686, 0.2706831, -0.5217806, 0.5185115

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8984203, upper bound: 1.8894747
time: 2.23 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8984203, upper bound: 1.8901655
time: 2.08 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.2565551, 0.9744402, -1.1795646, 1.1480556
1: -0.2987980, 0.3325009, -0.3448424, 0.3715456, -0.6703436, 0.6773433
2: -0.3846641, 0.3945331, -0.4289511, 0.4466715, -0.8313355, 0.8234842
3: -0.2706087, 0.2334781, -0.3015637, 0.2813208, -0.5519295, 0.5350419
4: -0.2944325, 0.3679506, -0.3472072, 0.4048828, -0.6993153, 0.7151577
5: -0.4267558, 0.4781152, -0.4669372, 0.5324135, -0.9591693, 0.9450524
6: -0.1026776, 1.2858521, -0.1975362, 1.3229021, -1.4255798, 1.4833882
7: -0.3313026, 0.4365475, -0.3851250, 0.4853222, -0.8166248, 0.8216725
8: -0.3483621, 0.3910702, -0.3923242, 0.4465339, -0.7948959, 0.7833945
9: -0.2218433, 0.2653263, -0.2674738, 0.3125915, -0.5344348, 0.5328001

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8984203, upper bound: 1.8892414
time: 1.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8984203, upper bound: 1.8898142
time: 1.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.2067769, 0.8844675, -1.0895919, 1.0982772
1: -0.2987980, 0.3325009, -0.2982824, 0.3353040, -0.6341020, 0.6307833
2: -0.3846641, 0.3945331, -0.3830329, 0.3976636, -0.7823277, 0.7775661
3: -0.2706087, 0.2334781, -0.2701186, 0.2365419, -0.5071506, 0.5035967
4: -0.2944325, 0.3679506, -0.2941514, 0.3709590, -0.6653916, 0.6621019
5: -0.4267558, 0.4781152, -0.4239522, 0.4835648, -0.9103206, 0.9020674
6: -0.1026776, 1.2858521, -0.0985513, 1.2864097, -1.3890874, 1.3844035
7: -0.3313026, 0.4365475, -0.3340706, 0.4360552, -0.7673578, 0.7706181
8: -0.3483621, 0.3910702, -0.3514939, 0.3950947, -0.7434567, 0.7425641
9: -0.2218433, 0.2653263, -0.2252686, 0.2706831, -0.4925264, 0.4905948

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8984203, upper bound: 1.8900493
time: 1.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8984203, upper bound: 1.8905728
time: 2.08 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -0.2382691, 0.9507946, -1.2073498, 1.2127093
1: -0.3448424, 0.3715456, -0.3297816, 0.3564569, -0.7012994, 0.7013272
2: -0.4289511, 0.4466715, -0.4150444, 0.4270655, -0.8560166, 0.8617158
3: -0.3015637, 0.2813208, -0.2916944, 0.2636827, -0.5652465, 0.5730152
4: -0.3472072, 0.4048828, -0.3300874, 0.3899961, -0.7372032, 0.7349701
5: -0.4669372, 0.5324135, -0.4548751, 0.5089113, -0.9758484, 0.9872885
6: -0.1975362, 1.3229021, -0.1680785, 1.3109281, -1.5084643, 1.4909806
7: -0.3851250, 0.4853222, -0.3655823, 0.4693472, -0.8544722, 0.8509045
8: -0.3923242, 0.4465339, -0.3765907, 0.4246833, -0.8170075, 0.8231246
9: -0.2674738, 0.3125915, -0.2510976, 0.2932430, -0.5607168, 0.5636891

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8890977, upper bound: 1.8886822
time: 2.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8894640
time: 2.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -0.2566187, 0.9745614, -1.2311165, 1.2310588
1: -0.3448424, 0.3715456, -0.3449113, 0.3714673, -0.7163098, 0.7164569
2: -0.4289511, 0.4466715, -0.4290012, 0.4467482, -0.8756993, 0.8756726
3: -0.3015637, 0.2813208, -0.3016051, 0.2813783, -0.5829420, 0.5829259
4: -0.3472072, 0.4048828, -0.3472668, 0.4047099, -0.7519171, 0.7521496
5: -0.4669372, 0.5324135, -0.4669960, 0.5325068, -0.9994440, 0.9994094
6: -0.1975362, 1.3229021, -0.1977043, 1.3230022, -1.5205383, 1.5206063
7: -0.3851250, 0.4853222, -0.3851508, 0.4853825, -0.8705075, 0.8704730
8: -0.3923242, 0.4465339, -0.3923964, 0.4462870, -0.8386112, 0.8389302
9: -0.2674738, 0.3125915, -0.2675011, 0.3124065, -0.5798803, 0.5800927

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8890977, upper bound: 1.8886822
time: 2.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8894640, upper bound: 1.8894640
time: 2.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2067769, 0.8844675, -0.2382691, 0.9507946, -1.1575714, 1.1227366
1: -0.2982824, 0.3353040, -0.3297816, 0.3564569, -0.6547394, 0.6650856
2: -0.3830329, 0.3976636, -0.4150444, 0.4270655, -0.8100984, 0.8127080
3: -0.2701186, 0.2365419, -0.2916944, 0.2636827, -0.5338013, 0.5282363
4: -0.2941514, 0.3709590, -0.3300874, 0.3899961, -0.6841474, 0.7010463
5: -0.4239522, 0.4835648, -0.4548751, 0.5089113, -0.9328635, 0.9384398
6: -0.0985513, 1.2864097, -0.1680785, 1.3109281, -1.4094794, 1.4544883
7: -0.3340706, 0.4360552, -0.3655823, 0.4693472, -0.8034178, 0.8016376
8: -0.3514939, 0.3950947, -0.3765907, 0.4246833, -0.7761772, 0.7716854
9: -0.2252686, 0.2706831, -0.2510976, 0.2932430, -0.5185115, 0.5217806

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8899036, upper bound: 1.8891703
time: 2.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8900230, upper bound: 1.8897896
time: 2.32 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2067769, 0.8844675, -0.2566187, 0.9745614, -1.1813383, 1.1410861
1: -0.2982824, 0.3353040, -0.3449113, 0.3714673, -0.6697497, 0.6802152
2: -0.3830329, 0.3976636, -0.4290012, 0.4467482, -0.8297811, 0.8266648
3: -0.2701186, 0.2365419, -0.3016051, 0.2813783, -0.5514969, 0.5381470
4: -0.2941514, 0.3709590, -0.3472668, 0.4047099, -0.6988612, 0.7182258
5: -0.4239522, 0.4835648, -0.4669960, 0.5325068, -0.9564590, 0.9505607
6: -0.0985513, 1.2864097, -0.1977043, 1.3230022, -1.4215536, 1.4841139
7: -0.3340706, 0.4360552, -0.3851508, 0.4853825, -0.8194531, 0.8212061
8: -0.3514939, 0.3950947, -0.3923964, 0.4462870, -0.7977809, 0.7874910
9: -0.2252686, 0.2706831, -0.2675011, 0.3124065, -0.5376750, 0.5381842

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8899036, upper bound: 1.8891703
time: 2.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8900230, upper bound: 1.8897896
time: 2.45 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -0.2051244, 0.8915004, -1.1480556, 1.1795646
1: -0.3448424, 0.3715456, -0.2987980, 0.3325009, -0.6773433, 0.6703436
2: -0.4289511, 0.4466715, -0.3846641, 0.3945331, -0.8234842, 0.8313355
3: -0.3015637, 0.2813208, -0.2706087, 0.2334781, -0.5350419, 0.5519295
4: -0.3472072, 0.4048828, -0.2944325, 0.3679506, -0.7151577, 0.6993153
5: -0.4669372, 0.5324135, -0.4267558, 0.4781152, -0.9450524, 0.9591693
6: -0.1975362, 1.3229021, -0.1026776, 1.2858521, -1.4833882, 1.4255798
7: -0.3851250, 0.4853222, -0.3313026, 0.4365475, -0.8216725, 0.8166248
8: -0.3923242, 0.4465339, -0.3483621, 0.3910702, -0.7833945, 0.7948959
9: -0.2674738, 0.3125915, -0.2218433, 0.2653263, -0.5328001, 0.5344348

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8894365, upper bound: 1.8892509
time: 2.22 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8897896, upper bound: 1.8900230
time: 2.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -0.2067769, 0.8844675, -1.1410227, 1.1812171
1: -0.3448424, 0.3715456, -0.2982824, 0.3353040, -0.6801464, 0.6698281
2: -0.4289511, 0.4466715, -0.3830329, 0.3976636, -0.8266147, 0.8297044
3: -0.3015637, 0.2813208, -0.2701186, 0.2365419, -0.5381056, 0.5514394
4: -0.3472072, 0.4048828, -0.2941514, 0.3709590, -0.7181662, 0.6990341
5: -0.4669372, 0.5324135, -0.4239522, 0.4835648, -0.9505020, 0.9563657
6: -0.1975362, 1.3229021, -0.0985513, 1.2864097, -1.4839458, 1.4214535
7: -0.3851250, 0.4853222, -0.3340706, 0.4360552, -0.8211802, 0.8193928
8: -0.3923242, 0.4465339, -0.3514939, 0.3950947, -0.7874188, 0.7980278
9: -0.2674738, 0.3125915, -0.2252686, 0.2706831, -0.5381569, 0.5378601

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8894365, upper bound: 1.8892509
time: 2.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8897896, upper bound: 1.8900230
time: 2.36 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2067769, 0.8844675, -0.2051244, 0.8915004, -1.0982772, 1.0895919
1: -0.2982824, 0.3353040, -0.2987980, 0.3325009, -0.6307833, 0.6341020
2: -0.3830329, 0.3976636, -0.3846641, 0.3945331, -0.7775661, 0.7823277
3: -0.2701186, 0.2365419, -0.2706087, 0.2334781, -0.5035967, 0.5071506
4: -0.2941514, 0.3709590, -0.2944325, 0.3679506, -0.6621019, 0.6653916
5: -0.4239522, 0.4835648, -0.4267558, 0.4781152, -0.9020674, 0.9103206
6: -0.0985513, 1.2864097, -0.1026776, 1.2858521, -1.3844035, 1.3890874
7: -0.3340706, 0.4360552, -0.3313026, 0.4365475, -0.7706181, 0.7673578
8: -0.3514939, 0.3950947, -0.3483621, 0.3910702, -0.7425641, 0.7434567
9: -0.2252686, 0.2706831, -0.2218433, 0.2653263, -0.4905948, 0.4925264

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8902531, upper bound: 1.8899819
time: 2.34 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8903516, upper bound: 1.8905455
time: 2.16 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2067769, 0.8844675, -0.2067769, 0.8844675, -1.0912443, 1.0912443
1: -0.2982824, 0.3353040, -0.2982824, 0.3353040, -0.6335864, 0.6335864
2: -0.3830329, 0.3976636, -0.3830329, 0.3976636, -0.7806965, 0.7806965
3: -0.2701186, 0.2365419, -0.2701186, 0.2365419, -0.5066605, 0.5066605
4: -0.2941514, 0.3709590, -0.2941514, 0.3709590, -0.6651103, 0.6651103
5: -0.4239522, 0.4835648, -0.4239522, 0.4835648, -0.9075170, 0.9075170
6: -0.0985513, 1.2864097, -0.0985513, 1.2864097, -1.3849611, 1.3849611
7: -0.3340706, 0.4360552, -0.3340706, 0.4360552, -0.7701259, 0.7701259
8: -0.3514939, 0.3950947, -0.3514939, 0.3950947, -0.7465885, 0.7465885
9: -0.2252686, 0.2706831, -0.2252686, 0.2706831, -0.4959517, 0.4959517

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8902531, upper bound: 1.8899819
time: 2.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8903516, upper bound: 1.8905455
time: 2.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.8261355, 1.6957741, -1.9340432, 1.7769301
1: -0.3297816, 0.3564569, -0.7845350, 0.7813374, -1.1111190, 1.1409919
2: -0.4150444, 0.4270655, -0.8624935, 1.0301049, -1.4451492, 1.2895589
3: -0.2916944, 0.2636827, -0.6554159, 0.7940063, -1.0857008, 0.9190986
4: -0.3300874, 0.3899961, -0.8107981, 1.0799718, -1.4100592, 1.2007942
5: -0.4548751, 0.5089113, -1.0105178, 1.1361258, -1.5910008, 1.5194291
6: -0.1680785, 1.3109281, -1.0676947, 1.6187458, -1.7868243, 2.3786228
7: -0.3655823, 0.4693472, -0.9711751, 0.9359220, -1.3015044, 1.4405223
8: -0.3765907, 0.4246833, -0.8822247, 1.1008432, -1.4774339, 1.3069080
9: -0.2510976, 0.2932430, -0.7688699, 0.8852291, -1.1363267, 1.0621129

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8815783, upper bound: 1.8236606
time: 1.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8246327
time: 2.20 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.8663305, 1.7437507, -1.9820198, 1.8171251
1: -0.3297816, 0.3564569, -0.8130708, 0.8073345, -1.1371162, 1.1695278
2: -0.4150444, 0.4270655, -0.8903426, 1.0682744, -1.4833188, 1.3174081
3: -0.2916944, 0.2636827, -0.6797470, 0.8258173, -1.1175117, 0.9434298
4: -0.3300874, 0.3899961, -0.8397647, 1.1246009, -1.4546883, 1.2297608
5: -0.4548751, 0.5089113, -1.0506577, 1.1777372, -1.6326122, 1.5595690
6: -0.1680785, 1.3109281, -1.1253142, 1.6426497, -1.8107282, 2.4362423
7: -0.3655823, 0.4693472, -1.0080607, 0.9664483, -1.3320307, 1.4774079
8: -0.3765907, 0.4246833, -0.9167264, 1.1443964, -1.5209872, 1.3414097
9: -0.2510976, 0.2932430, -0.7970768, 0.9224179, -1.1735156, 1.0903199

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8815783, upper bound: 1.8236606
time: 1.99 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8246327
time: 2.22 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.8261355, 1.6957741, -1.9008985, 1.7176359
1: -0.2987980, 0.3325009, -0.7845350, 0.7813374, -1.0801353, 1.1170359
2: -0.3846641, 0.3945331, -0.8624935, 1.0301049, -1.4147689, 1.2570266
3: -0.2706087, 0.2334781, -0.6554159, 0.7940063, -1.0646150, 0.8888940
4: -0.2944325, 0.3679506, -0.8107981, 1.0799718, -1.3744043, 1.1787486
5: -0.4267558, 0.4781152, -1.0105178, 1.1361258, -1.5628816, 1.4886330
6: -0.1026776, 1.2858521, -1.0676947, 1.6187458, -1.7214234, 2.3535466
7: -0.3313026, 0.4365475, -0.9711751, 0.9359220, -1.2672246, 1.4077227
8: -0.3483621, 0.3910702, -0.8822247, 1.1008432, -1.4492053, 1.2732949
9: -0.2218433, 0.2653263, -0.7688699, 0.8852291, -1.1070724, 1.0341961

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8815783, upper bound: 1.8240178
time: 2.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8248847
time: 1.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.8663305, 1.7437507, -1.9488751, 1.7578309
1: -0.2987980, 0.3325009, -0.8130708, 0.8073345, -1.1061325, 1.1455717
2: -0.3846641, 0.3945331, -0.8903426, 1.0682744, -1.4529384, 1.2848758
3: -0.2706087, 0.2334781, -0.6797470, 0.8258173, -1.0964260, 0.9132252
4: -0.2944325, 0.3679506, -0.8397647, 1.1246009, -1.4190334, 1.2077153
5: -0.4267558, 0.4781152, -1.0506577, 1.1777372, -1.6044930, 1.5287730
6: -0.1026776, 1.2858521, -1.1253142, 1.6426497, -1.7453272, 2.4111662
7: -0.3313026, 0.4365475, -1.0080607, 0.9664483, -1.2977508, 1.4446082
8: -0.3483621, 0.3910702, -0.9167264, 1.1443964, -1.4927585, 1.3077966
9: -0.2218433, 0.2653263, -0.7970768, 0.9224179, -1.1442612, 1.0624031

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8815783, upper bound: 1.8240178
time: 2.17 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8822187, upper bound: 1.8248847
time: 1.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.7844378, 1.6371753, -1.8754444, 1.7352324
1: -0.3297816, 0.3564569, -0.7545584, 0.7572792, -1.0870608, 1.1110153
2: -0.4150444, 0.4270655, -0.8325227, 0.9933141, -1.4083585, 1.2595882
3: -0.2916944, 0.2636827, -0.6313741, 0.7623650, -1.0540594, 0.8950568
4: -0.3300874, 0.3899961, -0.7793881, 1.0365195, -1.3666070, 1.1693841
5: -0.4548751, 0.5089113, -0.9667646, 1.1034745, -1.5583496, 1.4756758
6: -0.1680785, 1.3109281, -1.0031563, 1.5844967, -1.7525753, 2.3140845
7: -0.3655823, 0.4693472, -0.9345270, 0.9043884, -1.2699707, 1.4038742
8: -0.3765907, 0.4246833, -0.8537888, 1.0604813, -1.4370720, 1.2784722
9: -0.2510976, 0.2932430, -0.7392236, 0.8510032, -1.1021008, 1.0324667

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8896054, upper bound: 1.8301955
time: 2.47 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8898685, upper bound: 1.8313821
time: 2.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.7919115, 1.6434530, -1.8817221, 1.7427061
1: -0.3297816, 0.3564569, -0.7598349, 0.7628895, -1.0926712, 1.1162918
2: -0.4150444, 0.4270655, -0.8372648, 1.0011079, -1.4161522, 1.2643303
3: -0.2916944, 0.2636827, -0.6400046, 0.7686682, -1.0603626, 0.9036874
4: -0.3300874, 0.3899961, -0.7842556, 1.0456074, -1.3756948, 1.1742516
5: -0.4548751, 0.5089113, -0.9735982, 1.1139143, -1.5687892, 1.4825094
6: -0.1680785, 1.3109281, -1.0126145, 1.5860882, -1.7541667, 2.3235426
7: -0.3655823, 0.4693472, -0.9417365, 0.9100840, -1.2756664, 1.4110837
8: -0.3765907, 0.4246833, -0.8621641, 1.0699773, -1.4465680, 1.2868475
9: -0.2510976, 0.2932430, -0.7442783, 0.8592910, -1.1103885, 1.0375214

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8896054, upper bound: 1.8301955
time: 1.97 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8898685, upper bound: 1.8313821
time: 2.19 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.7844378, 1.6371753, -1.8422997, 1.6759381
1: -0.2987980, 0.3325009, -0.7545584, 0.7572792, -1.0560772, 1.0870593
2: -0.3846641, 0.3945331, -0.8325227, 0.9933141, -1.3779781, 1.2270559
3: -0.2706087, 0.2334781, -0.6313741, 0.7623650, -1.0329738, 0.8648522
4: -0.2944325, 0.3679506, -0.7793881, 1.0365195, -1.3309520, 1.1473386
5: -0.4267558, 0.4781152, -0.9667646, 1.1034745, -1.5302303, 1.4448798
6: -0.1026776, 1.2858521, -1.0031563, 1.5844967, -1.6871743, 2.2890084
7: -0.3313026, 0.4365475, -0.9345270, 0.9043884, -1.2356910, 1.3710746
8: -0.3483621, 0.3910702, -0.8537888, 1.0604813, -1.4088434, 1.2448590
9: -0.2218433, 0.2653263, -0.7392236, 0.8510032, -1.0728465, 1.0045499

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8846945, upper bound: 1.8311169
time: 2.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8850059, upper bound: 1.8319764
time: 1.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.7919115, 1.6434530, -1.8485774, 1.6834118
1: -0.2987980, 0.3325009, -0.7598349, 0.7628895, -1.0616875, 1.0923357
2: -0.3846641, 0.3945331, -0.8372648, 1.0011079, -1.3857720, 1.2317979
3: -0.2706087, 0.2334781, -0.6400046, 0.7686682, -1.0392768, 0.8734828
4: -0.2944325, 0.3679506, -0.7842556, 1.0456074, -1.3400400, 1.1522062
5: -0.4267558, 0.4781152, -0.9735982, 1.1139143, -1.5406700, 1.4517133
6: -0.1026776, 1.2858521, -1.0126145, 1.5860882, -1.6887658, 2.2984667
7: -0.3313026, 0.4365475, -0.9417365, 0.9100840, -1.2413865, 1.3782840
8: -0.3483621, 0.3910702, -0.8621641, 1.0699773, -1.4183394, 1.2532344
9: -0.2218433, 0.2653263, -0.7442783, 0.8592910, -1.0811342, 1.0096046

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8846945, upper bound: 1.8311169
time: 1.96 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8850059, upper bound: 1.8319764
time: 3.10 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -0.8261355, 1.6957741, -1.9523292, 1.8005757
1: -0.3448424, 0.3715456, -0.7845350, 0.7813374, -1.1261798, 1.1560806
2: -0.4289511, 0.4466715, -0.8624935, 1.0301049, -1.4590560, 1.3091649
3: -0.3015637, 0.2813208, -0.6554159, 0.7940063, -1.0955701, 0.9367367
4: -0.3472072, 0.4048828, -0.8107981, 1.0799718, -1.4271790, 1.2156808
5: -0.4669372, 0.5324135, -1.0105178, 1.1361258, -1.6030630, 1.5429313
6: -0.1975362, 1.3229021, -1.0676947, 1.6187458, -1.8162820, 2.3905969
7: -0.3851250, 0.4853222, -0.9711751, 0.9359220, -1.3210471, 1.4564973
8: -0.3923242, 0.4465339, -0.8822247, 1.1008432, -1.4931674, 1.3287585
9: -0.2674738, 0.3125915, -0.7688699, 0.8852291, -1.1527029, 1.0814614

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8752946, upper bound: 1.8234148
time: 1.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8245247
time: 1.98 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -0.8663305, 1.7437507, -2.0003059, 1.8407707
1: -0.3448424, 0.3715456, -0.8130708, 0.8073345, -1.1521770, 1.1846164
2: -0.4289511, 0.4466715, -0.8903426, 1.0682744, -1.4972255, 1.3370141
3: -0.3015637, 0.2813208, -0.6797470, 0.8258173, -1.1273811, 0.9610679
4: -0.3472072, 0.4048828, -0.8397647, 1.1246009, -1.4718081, 1.2446475
5: -0.4669372, 0.5324135, -1.0506577, 1.1777372, -1.6446744, 1.5830712
6: -0.1975362, 1.3229021, -1.1253142, 1.6426497, -1.8401859, 2.4482164
7: -0.3851250, 0.4853222, -1.0080607, 0.9664483, -1.3515732, 1.4933829
8: -0.3923242, 0.4465339, -0.9167264, 1.1443964, -1.5367206, 1.3632603
9: -0.2674738, 0.3125915, -0.7970768, 0.9224179, -1.1898918, 1.1096684

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8752946, upper bound: 1.8234148
time: 2.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8245247
time: 2.11 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2067769, 0.8844675, -0.8261355, 1.6957741, -1.9025509, 1.7106030
1: -0.2982824, 0.3353040, -0.7845350, 0.7813374, -1.0796199, 1.1198390
2: -0.3830329, 0.3976636, -0.8624935, 1.0301049, -1.4131378, 1.2601571
3: -0.2701186, 0.2365419, -0.6554159, 0.7940063, -1.0641249, 0.8919578
4: -0.2941514, 0.3709590, -0.8107981, 1.0799718, -1.3741231, 1.1817571
5: -0.4239522, 0.4835648, -1.0105178, 1.1361258, -1.5600780, 1.4940827
6: -0.0985513, 1.2864097, -1.0676947, 1.6187458, -1.7172971, 2.3541045
7: -0.3340706, 0.4360552, -0.9711751, 0.9359220, -1.2699926, 1.4072304
8: -0.3514939, 0.3950947, -0.8822247, 1.1008432, -1.4523370, 1.2773193
9: -0.2252686, 0.2706831, -0.7688699, 0.8852291, -1.1104977, 1.0395530

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8757061, upper bound: 1.8238978
time: 2.22 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8774045, upper bound: 1.8248551
time: 1.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2067769, 0.8844675, -0.8663305, 1.7437507, -1.9505275, 1.7507980
1: -0.2982824, 0.3353040, -0.8130708, 0.8073345, -1.1056170, 1.1483748
2: -0.3830329, 0.3976636, -0.8903426, 1.0682744, -1.4513073, 1.2880062
3: -0.2701186, 0.2365419, -0.6797470, 0.8258173, -1.0959359, 0.9162889
4: -0.2941514, 0.3709590, -0.8397647, 1.1246009, -1.4187522, 1.2107238
5: -0.4239522, 0.4835648, -1.0506577, 1.1777372, -1.6016895, 1.5342226
6: -0.0985513, 1.2864097, -1.1253142, 1.6426497, -1.7412009, 2.4117241
7: -0.3340706, 0.4360552, -1.0080607, 0.9664483, -1.3005190, 1.4441159
8: -0.3514939, 0.3950947, -0.9167264, 1.1443964, -1.4958903, 1.3118211
9: -0.2252686, 0.2706831, -0.7970768, 0.9224179, -1.1476865, 1.0677599

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8757061, upper bound: 1.8238978
time: 2.30 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8774045, upper bound: 1.8248551
time: 2.23 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -0.7844378, 1.6371753, -1.8937304, 1.7588780
1: -0.3448424, 0.3715456, -0.7545584, 0.7572792, -1.1021216, 1.1261040
2: -0.4289511, 0.4466715, -0.8325227, 0.9933141, -1.4222652, 1.2791942
3: -0.3015637, 0.2813208, -0.6313741, 0.7623650, -1.0639287, 0.9126949
4: -0.3472072, 0.4048828, -0.7793881, 1.0365195, -1.3837267, 1.1842709
5: -0.4669372, 0.5324135, -0.9667646, 1.1034745, -1.5704117, 1.4991781
6: -0.1975362, 1.3229021, -1.0031563, 1.5844967, -1.7820330, 2.3260584
7: -0.3851250, 0.4853222, -0.9345270, 0.9043884, -1.2895133, 1.4198492
8: -0.3923242, 0.4465339, -0.8537888, 1.0604813, -1.4528055, 1.3003227
9: -0.2674738, 0.3125915, -0.7392236, 0.8510032, -1.1184771, 1.0518152

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8821048, upper bound: 1.8296678
time: 1.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8831035, upper bound: 1.8312210
time: 2.13 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -0.7919115, 1.6434530, -1.9000082, 1.7663517
1: -0.3448424, 0.3715456, -0.7598349, 0.7628895, -1.1077319, 1.1313804
2: -0.4289511, 0.4466715, -0.8372648, 1.0011079, -1.4300591, 1.2839363
3: -0.3015637, 0.2813208, -0.6400046, 0.7686682, -1.0702319, 0.9213254
4: -0.3472072, 0.4048828, -0.7842556, 1.0456074, -1.3928146, 1.1891383
5: -0.4669372, 0.5324135, -0.9735982, 1.1139143, -1.5808514, 1.5060117
6: -0.1975362, 1.3229021, -1.0126145, 1.5860882, -1.7836244, 2.3355165
7: -0.3851250, 0.4853222, -0.9417365, 0.9100840, -1.2952089, 1.4270587
8: -0.3923242, 0.4465339, -0.8621641, 1.0699773, -1.4623015, 1.3086979
9: -0.2674738, 0.3125915, -0.7442783, 0.8592910, -1.1267648, 1.0568699

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8821048, upper bound: 1.8296678
time: 1.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8831035, upper bound: 1.8312210
time: 2.18 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2067769, 0.8844675, -0.7844378, 1.6371753, -1.8439522, 1.6689053
1: -0.2982824, 0.3353040, -0.7545584, 0.7572792, -1.0555615, 1.0898623
2: -0.3830329, 0.3976636, -0.8325227, 0.9933141, -1.3763471, 1.2301863
3: -0.2701186, 0.2365419, -0.6313741, 0.7623650, -1.0324836, 0.8679160
4: -0.2941514, 0.3709590, -0.7793881, 1.0365195, -1.3306708, 1.1503471
5: -0.4239522, 0.4835648, -0.9667646, 1.1034745, -1.5274267, 1.4503293
6: -0.0985513, 1.2864097, -1.0031563, 1.5844967, -1.6830480, 2.2895660
7: -0.3340706, 0.4360552, -0.9345270, 0.9043884, -1.2384590, 1.3705823
8: -0.3514939, 0.3950947, -0.8537888, 1.0604813, -1.4119751, 1.2488835
9: -0.2252686, 0.2706831, -0.7392236, 0.8510032, -1.0762718, 1.0099066

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8809026, upper bound: 1.8309926
time: 2.03 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8814824, upper bound: 1.8319511
time: 1.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2067769, 0.8844675, -0.7919115, 1.6434530, -1.8502299, 1.6763790
1: -0.2982824, 0.3353040, -0.7598349, 0.7628895, -1.0611720, 1.0951388
2: -0.3830329, 0.3976636, -0.8372648, 1.0011079, -1.3841408, 1.2349284
3: -0.2701186, 0.2365419, -0.6400046, 0.7686682, -1.0387868, 0.8765465
4: -0.2941514, 0.3709590, -0.7842556, 1.0456074, -1.3397589, 1.1552145
5: -0.4239522, 0.4835648, -0.9735982, 1.1139143, -1.5378665, 1.4571630
6: -0.0985513, 1.2864097, -1.0126145, 1.5860882, -1.6846395, 2.2990241
7: -0.3340706, 0.4360552, -0.9417365, 0.9100840, -1.2441547, 1.3777916
8: -0.3514939, 0.3950947, -0.8621641, 1.0699773, -1.4214711, 1.2572588
9: -0.2252686, 0.2706831, -0.7442783, 0.8592910, -1.0845596, 1.0149614

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8809026, upper bound: 1.8309926
time: 1.92 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8814824, upper bound: 1.8319511
time: 1.73 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.8261355, 1.6957741, -0.2382691, 0.9507946, -1.7769301, 1.9340432
1: -0.7845350, 0.7813374, -0.3297816, 0.3564569, -1.1409919, 1.1111190
2: -0.8624935, 1.0301049, -0.4150444, 0.4270655, -1.2895589, 1.4451492
3: -0.6554159, 0.7940063, -0.2916944, 0.2636827, -0.9190986, 1.0857008
4: -0.8107981, 1.0799718, -0.3300874, 0.3899961, -1.2007942, 1.4100592
5: -1.0105178, 1.1361258, -0.4548751, 0.5089113, -1.5194291, 1.5910008
6: -1.0676947, 1.6187458, -0.1680785, 1.3109281, -2.3786228, 1.7868243
7: -0.9711751, 0.9359220, -0.3655823, 0.4693472, -1.4405223, 1.3015044
8: -0.8822247, 1.1008432, -0.3765907, 0.4246833, -1.3069080, 1.4774339
9: -0.7688699, 0.8852291, -0.2510976, 0.2932430, -1.0621129, 1.1363267

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8485039, upper bound: 1.8919247
time: 2.33 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8480511, upper bound: 1.8920842
time: 2.32 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.8261355, 1.6957741, -0.2051244, 0.8915004, -1.7176359, 1.9008985
1: -0.7845350, 0.7813374, -0.2987980, 0.3325009, -1.1170359, 1.0801353
2: -0.8624935, 1.0301049, -0.3846641, 0.3945331, -1.2570266, 1.4147689
3: -0.6554159, 0.7940063, -0.2706087, 0.2334781, -0.8888940, 1.0646150
4: -0.8107981, 1.0799718, -0.2944325, 0.3679506, -1.1787486, 1.3744043
5: -1.0105178, 1.1361258, -0.4267558, 0.4781152, -1.4886330, 1.5628816
6: -1.0676947, 1.6187458, -0.1026776, 1.2858521, -2.3535466, 1.7214234
7: -0.9711751, 0.9359220, -0.3313026, 0.4365475, -1.4077227, 1.2672246
8: -0.8822247, 1.1008432, -0.3483621, 0.3910702, -1.2732949, 1.4492053
9: -0.7688699, 0.8852291, -0.2218433, 0.2653263, -1.0341961, 1.1070724

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8485039, upper bound: 1.8919247
time: 2.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8480511, upper bound: 1.8920842
time: 2.73 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.8663305, 1.7437507, -0.2382691, 0.9507946, -1.8171251, 1.9820198
1: -0.8130708, 0.8073345, -0.3297816, 0.3564569, -1.1695278, 1.1371162
2: -0.8903426, 1.0682744, -0.4150444, 0.4270655, -1.3174081, 1.4833188
3: -0.6797470, 0.8258173, -0.2916944, 0.2636827, -0.9434298, 1.1175117
4: -0.8397647, 1.1246009, -0.3300874, 0.3899961, -1.2297608, 1.4546883
5: -1.0506577, 1.1777372, -0.4548751, 0.5089113, -1.5595690, 1.6326122
6: -1.1253142, 1.6426497, -0.1680785, 1.3109281, -2.4362423, 1.8107282
7: -1.0080607, 0.9664483, -0.3655823, 0.4693472, -1.4774079, 1.3320307
8: -0.9167264, 1.1443964, -0.3765907, 0.4246833, -1.3414097, 1.5209872
9: -0.7970768, 0.9224179, -0.2510976, 0.2932430, -1.0903199, 1.1735156

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8245988, upper bound: 1.8813854
time: 3.01 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8245848, upper bound: 1.8817604
time: 2.19 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.8663305, 1.7437507, -0.2051244, 0.8915004, -1.7578309, 1.9488751
1: -0.8130708, 0.8073345, -0.2987980, 0.3325009, -1.1455717, 1.1061325
2: -0.8903426, 1.0682744, -0.3846641, 0.3945331, -1.2848758, 1.4529384
3: -0.6797470, 0.8258173, -0.2706087, 0.2334781, -0.9132252, 1.0964260
4: -0.8397647, 1.1246009, -0.2944325, 0.3679506, -1.2077153, 1.4190334
5: -1.0506577, 1.1777372, -0.4267558, 0.4781152, -1.5287730, 1.6044930
6: -1.1253142, 1.6426497, -0.1026776, 1.2858521, -2.4111662, 1.7453272
7: -1.0080607, 0.9664483, -0.3313026, 0.4365475, -1.4446082, 1.2977508
8: -0.9167264, 1.1443964, -0.3483621, 0.3910702, -1.3077966, 1.4927585
9: -0.7970768, 0.9224179, -0.2218433, 0.2653263, -1.0624031, 1.1442612

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8245988, upper bound: 1.8813854
time: 2.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8245848, upper bound: 1.8817604
time: 2.26 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.8261355, 1.6957741, -0.2565551, 0.9744402, -1.8005757, 1.9523292
1: -0.7845350, 0.7813374, -0.3448424, 0.3715456, -1.1560806, 1.1261798
2: -0.8624935, 1.0301049, -0.4289511, 0.4466715, -1.3091649, 1.4590560
3: -0.6554159, 0.7940063, -0.3015637, 0.2813208, -0.9367367, 1.0955701
4: -0.8107981, 1.0799718, -0.3472072, 0.4048828, -1.2156808, 1.4271790
5: -1.0105178, 1.1361258, -0.4669372, 0.5324135, -1.5429313, 1.6030630
6: -1.0676947, 1.6187458, -0.1975362, 1.3229021, -2.3905969, 1.8162820
7: -0.9711751, 0.9359220, -0.3851250, 0.4853222, -1.4564973, 1.3210471
8: -0.8822247, 1.1008432, -0.3923242, 0.4465339, -1.3287585, 1.4931674
9: -0.7688699, 0.8852291, -0.2674738, 0.3125915, -1.0814614, 1.1527029

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8473310, upper bound: 1.8819156
time: 2.12 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8469008, upper bound: 1.8820511
time: 1.93 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.8261355, 1.6957741, -0.2067769, 0.8844675, -1.7106030, 1.9025509
1: -0.7845350, 0.7813374, -0.2982824, 0.3353040, -1.1198390, 1.0796199
2: -0.8624935, 1.0301049, -0.3830329, 0.3976636, -1.2601571, 1.4131378
3: -0.6554159, 0.7940063, -0.2701186, 0.2365419, -0.8919578, 1.0641249
4: -0.8107981, 1.0799718, -0.2941514, 0.3709590, -1.1817571, 1.3741231
5: -1.0105178, 1.1361258, -0.4239522, 0.4835648, -1.4940827, 1.5600780
6: -1.0676947, 1.6187458, -0.0985513, 1.2864097, -2.3541045, 1.7172971
7: -0.9711751, 0.9359220, -0.3340706, 0.4360552, -1.4072304, 1.2699926
8: -0.8822247, 1.1008432, -0.3514939, 0.3950947, -1.2773193, 1.4523370
9: -0.7688699, 0.8852291, -0.2252686, 0.2706831, -1.0395530, 1.1104977

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8473310, upper bound: 1.8822374
time: 1.76 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8469008, upper bound: 1.8823484
time: 2.33 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.8663305, 1.7437507, -0.2565551, 0.9744402, -1.8407707, 2.0003059
1: -0.8130708, 0.8073345, -0.3448424, 0.3715456, -1.1846164, 1.1521770
2: -0.8903426, 1.0682744, -0.4289511, 0.4466715, -1.3370141, 1.4972255
3: -0.6797470, 0.8258173, -0.3015637, 0.2813208, -0.9610679, 1.1273811
4: -0.8397647, 1.1246009, -0.3472072, 0.4048828, -1.2446475, 1.4718081
5: -1.0506577, 1.1777372, -0.4669372, 0.5324135, -1.5830712, 1.6446744
6: -1.1253142, 1.6426497, -0.1975362, 1.3229021, -2.4482164, 1.8401859
7: -1.0080607, 0.9664483, -0.3851250, 0.4853222, -1.4933829, 1.3515732
8: -0.9167264, 1.1443964, -0.3923242, 0.4465339, -1.3632603, 1.5367206
9: -0.7970768, 0.9224179, -0.2674738, 0.3125915, -1.1096684, 1.1898918

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8245027, upper bound: 1.8764897
time: 2.19 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8244919, upper bound: 1.8770641
time: 2.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.8663305, 1.7437507, -0.2067769, 0.8844675, -1.7507980, 1.9505275
1: -0.8130708, 0.8073345, -0.2982824, 0.3353040, -1.1483748, 1.1056170
2: -0.8903426, 1.0682744, -0.3830329, 0.3976636, -1.2880062, 1.4513073
3: -0.6797470, 0.8258173, -0.2701186, 0.2365419, -0.9162889, 1.0959359
4: -0.8397647, 1.1246009, -0.2941514, 0.3709590, -1.2107238, 1.4187522
5: -1.0506577, 1.1777372, -0.4239522, 0.4835648, -1.5342226, 1.6016895
6: -1.1253142, 1.6426497, -0.0985513, 1.2864097, -2.4117241, 1.7412009
7: -1.0080607, 0.9664483, -0.3340706, 0.4360552, -1.4441159, 1.3005190
8: -0.9167264, 1.1443964, -0.3514939, 0.3950947, -1.3118211, 1.4958903
9: -0.7970768, 0.9224179, -0.2252686, 0.2706831, -1.0677599, 1.1476865

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8245027, upper bound: 1.8766642
time: 2.24 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8244919, upper bound: 1.8771940
time: 2.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.7844378, 1.6371753, -0.2382691, 0.9507946, -1.7352324, 1.8754444
1: -0.7545584, 0.7572792, -0.3297816, 0.3564569, -1.1110153, 1.0870608
2: -0.8325227, 0.9933141, -0.4150444, 0.4270655, -1.2595882, 1.4083585
3: -0.6313741, 0.7623650, -0.2916944, 0.2636827, -0.8950568, 1.0540594
4: -0.7793881, 1.0365195, -0.3300874, 0.3899961, -1.1693841, 1.3666070
5: -0.9667646, 1.1034745, -0.4548751, 0.5089113, -1.4756758, 1.5583496
6: -1.0031563, 1.5844967, -0.1680785, 1.3109281, -2.3140845, 1.7525753
7: -0.9345270, 0.9043884, -0.3655823, 0.4693472, -1.4038742, 1.2699707
8: -0.8537888, 1.0604813, -0.3765907, 0.4246833, -1.2784722, 1.4370720
9: -0.7392236, 0.8510032, -0.2510976, 0.2932430, -1.0324667, 1.1021008

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 168

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8423760, upper bound: 1.8880195
time: 3.90 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8494964, upper bound: 1.8946857
time: 2.91 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.7844378, 1.6371753, -0.2051244, 0.8915004, -1.6759381, 1.8422997
1: -0.7545584, 0.7572792, -0.2987980, 0.3325009, -1.0870593, 1.0560772
2: -0.8325227, 0.9933141, -0.3846641, 0.3945331, -1.2270559, 1.3779781
3: -0.6313741, 0.7623650, -0.2706087, 0.2334781, -0.8648522, 1.0329738
4: -0.7793881, 1.0365195, -0.2944325, 0.3679506, -1.1473386, 1.3309520
5: -0.9667646, 1.1034745, -0.4267558, 0.4781152, -1.4448798, 1.5302303
6: -1.0031563, 1.5844967, -0.1026776, 1.2858521, -2.2890084, 1.6871743
7: -0.9345270, 0.9043884, -0.3313026, 0.4365475, -1.3710746, 1.2356910
8: -0.8537888, 1.0604813, -0.3483621, 0.3910702, -1.2448590, 1.4088434
9: -0.7392236, 0.8510032, -0.2218433, 0.2653263, -1.0045499, 1.0728465

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 168

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8423760, upper bound: 1.8880428
time: 2.49 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8494964, upper bound: 1.8946857
time: 2.44 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.7919115, 1.6434530, -0.2382691, 0.9507946, -1.7427061, 1.8817221
1: -0.7598349, 0.7628895, -0.3297816, 0.3564569, -1.1162918, 1.0926712
2: -0.8372648, 1.0011079, -0.4150444, 0.4270655, -1.2643303, 1.4161522
3: -0.6400046, 0.7686682, -0.2916944, 0.2636827, -0.9036874, 1.0603626
4: -0.7842556, 1.0456074, -0.3300874, 0.3899961, -1.1742516, 1.3756948
5: -0.9735982, 1.1139143, -0.4548751, 0.5089113, -1.4825094, 1.5687892
6: -1.0126145, 1.5860882, -0.1680785, 1.3109281, -2.3235426, 1.7541667
7: -0.9417365, 0.9100840, -0.3655823, 0.4693472, -1.4110837, 1.2756664
8: -0.8621641, 1.0699773, -0.3765907, 0.4246833, -1.2868475, 1.4465680
9: -0.7442783, 0.8592910, -0.2510976, 0.2932430, -1.0375214, 1.1103885

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 168

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8171004, upper bound: 1.8735741
time: 2.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8313821, upper bound: 1.8898685
time: 2.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.7919115, 1.6434530, -0.2051244, 0.8915004, -1.6834118, 1.8485774
1: -0.7598349, 0.7628895, -0.2987980, 0.3325009, -1.0923357, 1.0616875
2: -0.8372648, 1.0011079, -0.3846641, 0.3945331, -1.2317979, 1.3857720
3: -0.6400046, 0.7686682, -0.2706087, 0.2334781, -0.8734828, 1.0392768
4: -0.7842556, 1.0456074, -0.2944325, 0.3679506, -1.1522062, 1.3400400
5: -0.9735982, 1.1139143, -0.4267558, 0.4781152, -1.4517133, 1.5406700
6: -1.0126145, 1.5860882, -0.1026776, 1.2858521, -2.2984667, 1.6887658
7: -0.9417365, 0.9100840, -0.3313026, 0.4365475, -1.3782840, 1.2413865
8: -0.8621641, 1.0699773, -0.3483621, 0.3910702, -1.2532344, 1.4183394
9: -0.7442783, 0.8592910, -0.2218433, 0.2653263, -1.0096046, 1.0811342

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 168

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8171004, upper bound: 1.8751209
time: 2.03 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8313821, upper bound: 1.8900152
time: 2.60 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.7844378, 1.6371753, -0.2565551, 0.9744402, -1.7588780, 1.8937304
1: -0.7545584, 0.7572792, -0.3448424, 0.3715456, -1.1261040, 1.1021216
2: -0.8325227, 0.9933141, -0.4289511, 0.4466715, -1.2791942, 1.4222652
3: -0.6313741, 0.7623650, -0.3015637, 0.2813208, -0.9126949, 1.0639287
4: -0.7793881, 1.0365195, -0.3472072, 0.4048828, -1.1842709, 1.3837267
5: -0.9667646, 1.1034745, -0.4669372, 0.5324135, -1.4991781, 1.5704117
6: -1.0031563, 1.5844967, -0.1975362, 1.3229021, -2.3260584, 1.7820330
7: -0.9345270, 0.9043884, -0.3851250, 0.4853222, -1.4198492, 1.2895133
8: -0.8537888, 1.0604813, -0.3923242, 0.4465339, -1.3003227, 1.4528055
9: -0.7392236, 0.8510032, -0.2674738, 0.3125915, -1.0518152, 1.1184771

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 168

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8403797, upper bound: 1.8746120
time: 1.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8480956, upper bound: 1.8850170
time: 2.23 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.7844378, 1.6371753, -0.2067769, 0.8844675, -1.6689053, 1.8439522
1: -0.7545584, 0.7572792, -0.2982824, 0.3353040, -1.0898623, 1.0555615
2: -0.8325227, 0.9933141, -0.3830329, 0.3976636, -1.2301863, 1.3763471
3: -0.6313741, 0.7623650, -0.2701186, 0.2365419, -0.8679160, 1.0324836
4: -0.7793881, 1.0365195, -0.2941514, 0.3709590, -1.1503471, 1.3306708
5: -0.9667646, 1.1034745, -0.4239522, 0.4835648, -1.4503293, 1.5274267
6: -1.0031563, 1.5844967, -0.0985513, 1.2864097, -2.2895660, 1.6830480
7: -0.9345270, 0.9043884, -0.3340706, 0.4360552, -1.3705823, 1.2384590
8: -0.8537888, 1.0604813, -0.3514939, 0.3950947, -1.2488835, 1.4119751
9: -0.7392236, 0.8510032, -0.2252686, 0.2706831, -1.0099066, 1.0762718

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 168

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8403797, upper bound: 1.8771146
time: 1.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8480956, upper bound: 1.8861910
time: 1.95 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.7919115, 1.6434530, -0.2565551, 0.9744402, -1.7663517, 1.9000082
1: -0.7598349, 0.7628895, -0.3448424, 0.3715456, -1.1313804, 1.1077319
2: -0.8372648, 1.0011079, -0.4289511, 0.4466715, -1.2839363, 1.4300591
3: -0.6400046, 0.7686682, -0.3015637, 0.2813208, -0.9213254, 1.0702319
4: -0.7842556, 1.0456074, -0.3472072, 0.4048828, -1.1891383, 1.3928146
5: -0.9735982, 1.1139143, -0.4669372, 0.5324135, -1.5060117, 1.5808514
6: -1.0126145, 1.5860882, -0.1975362, 1.3229021, -2.3355165, 1.7836244
7: -0.9417365, 0.9100840, -0.3851250, 0.4853222, -1.4270587, 1.2952089
8: -0.8621641, 1.0699773, -0.3923242, 0.4465339, -1.3086979, 1.4623015
9: -0.7442783, 0.8592910, -0.2674738, 0.3125915, -1.0568699, 1.1267648

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 168

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8167881, upper bound: 1.8674116
time: 2.47 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8312210, upper bound: 1.8831035
time: 2.22 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.7919115, 1.6434530, -0.2067769, 0.8844675, -1.6763790, 1.8502299
1: -0.7598349, 0.7628895, -0.2982824, 0.3353040, -1.0951388, 1.0611720
2: -0.8372648, 1.0011079, -0.3830329, 0.3976636, -1.2349284, 1.3841408
3: -0.6400046, 0.7686682, -0.2701186, 0.2365419, -0.8765465, 1.0387868
4: -0.7842556, 1.0456074, -0.2941514, 0.3709590, -1.1552145, 1.3397589
5: -0.9735982, 1.1139143, -0.4239522, 0.4835648, -1.4571630, 1.5378665
6: -1.0126145, 1.5860882, -0.0985513, 1.2864097, -2.2990241, 1.6846395
7: -0.9417365, 0.9100840, -0.3340706, 0.4360552, -1.3777916, 1.2441547
8: -0.8621641, 1.0699773, -0.3514939, 0.3950947, -1.2572588, 1.4214711
9: -0.7442783, 0.8592910, -0.2252686, 0.2706831, -1.0149614, 1.0845596

Time for backsubstitution: 1.53 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.094357967376709
rel_dist={6: [-1.9135725282356488, 1.9135718444666665]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 189

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8916922, upper bound: 1.8639348
time: 3.44 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8637098, upper bound: 1.8637097
time: 2.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 6.34 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 6.34
Output dim: 6, lower bound: -1.8916922, upper bound: 1.8639348
IS_A2, status: Status.UNKNOWN, split count: 1, time: 6.34
Output dim: 6, lower bound: -1.8637098, upper bound: 1.8637097

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.4283954, 1.2709696, -0.4712906, 1.3211114, -1.7495068, 1.7422602
1: -0.5065504, 0.5120057, -0.5395136, 0.5422920, -1.0488424, 1.0515194
2: -0.5936686, 0.6343641, -0.6257302, 0.6802884, -1.2739570, 1.2600943
3: -0.4099944, 0.4626325, -0.4393560, 0.5000688, -0.9100632, 0.9019885
4: -0.5260350, 0.5839709, -0.5584180, 0.6426379, -1.1686729, 1.1423889
5: -0.6253715, 0.7425824, -0.6672549, 0.7914629, -1.4168344, 1.4098372
6: -0.5344308, 1.4685735, -0.5977826, 1.4869769, -2.0214076, 2.0663562
7: -0.5838713, 0.6439485, -0.6288721, 0.6773571, -1.2612284, 1.2728206
8: -0.5708863, 0.6576684, -0.6064137, 0.7106374, -1.2815237, 1.2640821
9: -0.4413340, 0.4900914, -0.4792553, 0.5364693, -0.9778033, 0.9693466

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 189

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8637097, upper bound: 1.8637098
time: 2.22 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8637097, upper bound: 1.8637097
time: 1.88 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.1578722, 2.0902753, -0.4328149, 1.2749331, -2.4328053, 2.5230901
1: -1.0186968, 0.9981120, -0.5099469, 0.5155180, -1.5342147, 1.5080589
2: -1.0924278, 1.3473158, -0.5966854, 0.6399475, -1.7323754, 1.9440012
3: -0.8712537, 1.0596817, -0.4131823, 0.4670498, -1.3383034, 1.4728639
4: -1.0514562, 1.4630030, -0.5292383, 0.5912334, -1.6426896, 1.9922414
5: -1.3404843, 1.4613965, -0.6290373, 0.7492172, -2.0897014, 2.0904336
6: -1.5363673, 1.8078189, -0.5402350, 1.4686012, -3.0049686, 2.3480539
7: -1.2797452, 1.1851227, -0.5892617, 0.6470501, -1.9267954, 1.7743844
8: -1.1523812, 1.4595408, -0.5743003, 0.6642862, -1.8166673, 2.0338411
9: -1.0169351, 1.1958456, -0.4460403, 0.4957484, -1.5126835, 1.6418859

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8297824, upper bound: 1.8434120
time: 2.29 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8297503, upper bound: 1.8297503
time: 2.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 6.33 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 6.33
Output dim: 6, lower bound: -1.8637097, upper bound: 1.8637098
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 6.33
Output dim: 6, lower bound: -1.8637097, upper bound: 1.8637097
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 6.33
Output dim: 6, lower bound: -1.8297824, upper bound: 1.8434120
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 6.33
Output dim: 6, lower bound: -1.8297503, upper bound: 1.8297503

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.4283954, 1.2709696, -0.4283954, 1.2709696, -1.6993650, 1.6993650
1: -0.5065504, 0.5120057, -0.5065504, 0.5120057, -1.0185561, 1.0185561
2: -0.5936686, 0.6343641, -0.5936686, 0.6343641, -1.2280327, 1.2280327
3: -0.4099944, 0.4626325, -0.4099944, 0.4626325, -0.8726270, 0.8726270
4: -0.5260350, 0.5839709, -0.5260350, 0.5839709, -1.1100059, 1.1100059
5: -0.6253715, 0.7425824, -0.6253715, 0.7425824, -1.3679538, 1.3679538
6: -0.5344308, 1.4685735, -0.5344308, 1.4685735, -2.0030043, 2.0030043
7: -0.5838713, 0.6439485, -0.5838713, 0.6439485, -1.2278198, 1.2278198
8: -0.5708863, 0.6576684, -0.5708863, 0.6576684, -1.2285547, 1.2285547
9: -0.4413340, 0.4900914, -0.4413340, 0.4900914, -0.9314255, 0.9314255

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
time: 2.41 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8612994, upper bound: 1.8300520
time: 2.98 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.4283954, 1.2709696, -1.1578722, 2.0902753, -2.5186706, 2.4288418
1: -0.5065504, 0.5120057, -1.0186968, 0.9981120, -1.5046624, 1.5307025
2: -0.5936686, 0.6343641, -1.0924278, 1.3473158, -1.9409844, 1.7267920
3: -0.4099944, 0.4626325, -0.8712537, 1.0596817, -1.4696760, 1.3338861
4: -0.5260350, 0.5839709, -1.0514562, 1.4630030, -1.9890380, 1.6354271
5: -0.6253715, 0.7425824, -1.3404843, 1.4613965, -2.0867679, 2.0830667
6: -0.5344308, 1.4685735, -1.5363673, 1.8078189, -2.3422496, 3.0049407
7: -0.5838713, 0.6439485, -1.2797452, 1.1851227, -1.7689941, 1.9236937
8: -0.5708863, 0.6576684, -1.1523812, 1.4595408, -2.0304272, 1.8100495
9: -0.4413340, 0.4900914, -1.0169351, 1.1958456, -1.6371796, 1.5070266

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
time: 2.94 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8612994, upper bound: 1.8300520
time: 2.70 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -1.0812174, 1.9948463, -0.2366522, 0.9295844, -2.0108018, 2.2314985
1: -0.9644838, 0.9490871, -0.3239092, 0.3582276, -1.3227113, 1.2729963
2: -1.0389121, 1.2750793, -0.4077712, 0.4277734, -1.4666855, 1.6828505
3: -0.8224984, 0.9989403, -0.2875796, 0.2644011, -1.0868995, 1.2865199
4: -0.9954811, 1.3762964, -0.3242272, 0.3923352, -1.3878163, 1.7005236
5: -1.2634481, 1.3885615, -0.4463893, 0.5146791, -1.7781272, 1.8349508
6: -1.4254749, 1.7591095, -0.1502624, 1.2974494, -2.7229242, 1.9093719
7: -1.2092671, 1.1274813, -0.3649695, 0.4640141, -1.6732813, 1.4924508
8: -1.0914942, 1.3781033, -0.3759922, 0.4270988, -1.5185931, 1.7540956
9: -0.9604561, 1.1256905, -0.2524602, 0.2984439, -1.2588999, 1.3781506

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8226196, upper bound: 1.8371964
time: 2.41 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8297824, upper bound: 1.8434084
time: 2.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.41 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.41
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.41
Output dim: 6, lower bound: -1.8612994, upper bound: 1.8300520
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.41
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.41
Output dim: 6, lower bound: -1.8612994, upper bound: 1.8300520
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 6.41
Output dim: 6, lower bound: -1.8226196, upper bound: 1.8371964
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.41
Output dim: 6, lower bound: -1.8297824, upper bound: 1.8434084

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2336920, 0.9276679, -0.3746356, 1.1774944, -1.4111865, 1.3023034
1: -0.3219900, 0.3554728, -0.4537061, 0.4721467, -0.7941367, 0.8091789
2: -0.4061271, 0.4243887, -0.5438825, 0.5677086, -0.9738357, 0.9682713
3: -0.2861868, 0.2612684, -0.3714705, 0.4082282, -0.6944150, 0.6327389
4: -0.3218213, 0.3896247, -0.4731737, 0.5101413, -0.8319625, 0.8627985
5: -0.4451328, 0.5103453, -0.5660005, 0.6785762, -1.1237091, 1.0763458
6: -0.1471975, 1.2969497, -0.4270898, 1.4234610, -1.5706584, 1.7240396
7: -0.3616745, 0.4618227, -0.5211052, 0.5955039, -0.9571784, 0.9829279
8: -0.3729174, 0.4232192, -0.5129892, 0.5870273, -0.9599447, 0.9362084
9: -0.2493206, 0.2945979, -0.3872009, 0.4303454, -0.6796660, 0.6817988

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8941753, upper bound: 1.8876799
time: 3.13 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8957738, upper bound: 1.8884073
time: 2.28 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2360516, 0.9213381, -0.3122948, 1.0620886, -1.2981402, 1.2336329
1: -0.3221140, 0.3588542, -0.3938192, 0.4169319, -0.7390459, 0.7526734
2: -0.4051805, 0.4282180, -0.4784468, 0.5040495, -0.9092300, 0.9066648
3: -0.2861544, 0.2650071, -0.3342681, 0.3354221, -0.6215765, 0.5992752
4: -0.3222730, 0.3931868, -0.4034812, 0.4477319, -0.7700049, 0.7966680
5: -0.4428855, 0.5166113, -0.5110561, 0.5988473, -1.0417328, 1.0276673
6: -0.1440104, 1.2948549, -0.2988605, 1.3675743, -1.5115848, 1.5937154
7: -0.3651637, 0.4619819, -0.4451625, 0.5368371, -0.9020008, 0.9071444
8: -0.3766791, 0.4280382, -0.4452599, 0.5112426, -0.8879217, 0.8732980
9: -0.2533768, 0.3006897, -0.3198538, 0.3671188, -0.6204957, 0.6205435

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8877824, upper bound: 1.8876672
time: 2.99 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8883936, upper bound: 1.8883936
time: 2.67 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2336920, 0.9276679, -1.0812174, 1.9948463, -2.2285383, 2.0088854
1: -0.3219900, 0.3554728, -0.9644838, 0.9490871, -1.2710772, 1.3199565
2: -0.4061271, 0.4243887, -1.0389121, 1.2750793, -1.6812063, 1.4633008
3: -0.2861868, 0.2612684, -0.8224984, 0.9989403, -1.2851270, 1.0837668
4: -0.3218213, 0.3896247, -0.9954811, 1.3762964, -1.6981177, 1.3851058
5: -0.4451328, 0.5103453, -1.2634481, 1.3885615, -1.8336943, 1.7737935
6: -0.1471975, 1.2969497, -1.4254749, 1.7591095, -1.9063070, 2.7224245
7: -0.3616745, 0.4618227, -1.2092671, 1.1274813, -1.4891558, 1.6710899
8: -0.3729174, 0.4232192, -1.0914942, 1.3781033, -1.7510207, 1.5147134
9: -0.2493206, 0.2945979, -0.9604561, 1.1256905, -1.3750111, 1.2550540

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 122

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8553046, upper bound: 1.8229575
time: 2.56 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
time: 2.27 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2360516, 0.9213381, -0.9726731, 1.8678806, -2.1039321, 1.8940113
1: -0.3221140, 0.3588542, -0.8880665, 0.8779152, -1.2000291, 1.2469208
2: -0.4051805, 0.4282180, -0.9641317, 1.1708324, -1.5760128, 1.3923497
3: -0.2861544, 0.2650071, -0.7513143, 0.9114904, -1.1976448, 1.0163214
4: -0.3222730, 0.3931868, -0.9167696, 1.2493013, -1.5715743, 1.3099563
5: -0.4428855, 0.5166113, -1.1558347, 1.2834076, -1.7262931, 1.6724460
6: -0.1440104, 1.2948549, -1.2737988, 1.7003371, -1.8443475, 2.5686536
7: -0.3651637, 0.4619819, -1.1076499, 1.0464067, -1.4115704, 1.5696318
8: -0.3766791, 0.4280382, -1.0042189, 1.2607564, -1.6374354, 1.4322571
9: -0.2533768, 0.3006897, -0.8775474, 1.0233724, -1.2767493, 1.1782371

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 168

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8523301, upper bound: 1.8229489
time: 2.35 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8612994, upper bound: 1.8300520
time: 2.91 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.0297801, 1.9405415, -0.2342253, 0.9265025, -1.9562826, 2.1747668
1: -0.9285418, 0.9139240, -0.3219297, 0.3562762, -1.2848181, 1.2358537
2: -1.0042973, 1.2240520, -0.4059371, 0.4252372, -1.4295344, 1.6299890
3: -0.7857525, 0.9563106, -0.2862520, 0.2620396, -1.0477922, 1.2425625
4: -0.9586678, 1.3129430, -0.3218925, 0.3904935, -1.3491613, 1.6348355
5: -1.2135394, 1.3362877, -0.4447991, 0.5119392, -1.7254785, 1.7810868
6: -1.3574235, 1.7393484, -0.1464825, 1.2964524, -2.6538758, 1.8858309
7: -1.1596227, 1.0893941, -0.3623884, 0.4618666, -1.6214893, 1.4517825
8: -1.0486935, 1.3206129, -0.3739045, 0.4243667, -1.4730603, 1.6945174
9: -0.9191711, 1.0746045, -0.2501262, 0.2959566, -1.2151277, 1.3247308

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8286572, upper bound: 1.8412604
time: 3.06 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8286572, upper bound: 1.8434084
time: 2.94 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 7.35 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 6, lower bound: -1.8941753, upper bound: 1.8876799
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 6, lower bound: -1.8957738, upper bound: 1.8884073
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 6, lower bound: -1.8877824, upper bound: 1.8876672
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 6, lower bound: -1.8883936, upper bound: 1.8883936
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 6, lower bound: -1.8553046, upper bound: 1.8229575
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 6, lower bound: -1.8523301, upper bound: 1.8229489
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 6, lower bound: -1.8612994, upper bound: 1.8300520
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 6, lower bound: -1.8286572, upper bound: 1.8412604
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 6, lower bound: -1.8286572, upper bound: 1.8434084

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1897601, 0.8728909, -0.3804277, 1.2064865, -1.3962466, 1.2533185
1: -0.2865112, 0.3199377, -0.4633941, 0.4745054, -0.7610166, 0.7833319
2: -0.3732556, 0.3783560, -0.5554780, 0.5716958, -0.9449514, 0.9338340
3: -0.2622748, 0.2185330, -0.3783262, 0.4129742, -0.6752490, 0.5968592
4: -0.2799808, 0.3558938, -0.4840003, 0.5124459, -0.7924267, 0.8398942
5: -0.4171091, 0.4598510, -0.5774873, 0.6789486, -1.0960577, 1.0373384
6: -0.0791348, 1.2796773, -0.4533845, 1.4478763, -1.5270112, 1.7330618
7: -0.3149754, 0.4231526, -0.5270098, 0.6050475, -0.9200228, 0.9501624
8: -0.3345000, 0.3732905, -0.5269501, 0.5902122, -0.9247122, 0.9002406
9: -0.2072807, 0.2491801, -0.3914892, 0.4306548, -0.6379355, 0.6406693

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8941367, upper bound: 1.8869976
time: 2.45 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8941368, upper bound: 1.8876707
time: 2.44 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2312664, 0.9245848, -0.3461125, 1.1314895, -1.3627559, 1.2706974
1: -0.3200167, 0.3535223, -0.4278438, 0.4452750, -0.7652917, 0.7813661
2: -0.4043038, 0.4218536, -0.5159276, 0.5378709, -0.9421748, 0.9377812
3: -0.2848636, 0.2589081, -0.3558165, 0.3727614, -0.6576251, 0.6147246
4: -0.3194896, 0.3877837, -0.4422640, 0.4765136, -0.7960032, 0.8300476
5: -0.4435607, 0.5076071, -0.5435375, 0.6401211, -1.0836818, 1.0511446
6: -0.1434163, 1.2959421, -0.3732235, 1.4062937, -1.5497100, 1.6691656
7: -0.3590941, 0.4596763, -0.4846156, 0.5699029, -0.9289970, 0.9442919
8: -0.3708312, 0.4204896, -0.4833164, 0.5512007, -0.9220319, 0.9038060
9: -0.2469876, 0.2921115, -0.3532256, 0.3990798, -0.6460673, 0.6453371

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8956920, upper bound: 1.8879055
time: 2.15 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8956920, upper bound: 1.8884073
time: 2.14 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1924441, 0.8670551, -0.3234297, 1.1010146, -1.2934587, 1.1904848
1: -0.2868274, 0.3235715, -0.4087837, 0.4233904, -0.7102178, 0.7323551
2: -0.3723840, 0.3825782, -0.4957850, 0.5131722, -0.8855562, 0.8783631
3: -0.2623073, 0.2225588, -0.3444104, 0.3452059, -0.6075132, 0.5669692
4: -0.2806364, 0.3597653, -0.4199743, 0.4530167, -0.7336531, 0.7797396
5: -0.4149271, 0.4666748, -0.5277154, 0.6054564, -1.0203834, 0.9943902
6: -0.0768080, 1.2816662, -0.3362368, 1.3972590, -1.4740670, 1.6179030
7: -0.3188342, 0.4235124, -0.4565795, 0.5513378, -0.8701721, 0.8800919
8: -0.3386916, 0.3786046, -0.4642868, 0.5206126, -0.8593042, 0.8428914
9: -0.2115701, 0.2556143, -0.3281225, 0.3720302, -0.5836003, 0.5837368

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8875684, upper bound: 1.8869520
time: 2.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8877206, upper bound: 1.8876575
time: 2.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2335326, 0.9181919, -0.2839092, 1.0218475, -1.2553802, 1.2021011
1: -0.3200645, 0.3568251, -0.3700308, 0.3923557, -0.7124201, 0.7268559
2: -0.4032812, 0.4255934, -0.4536291, 0.4743348, -0.8776159, 0.8792225
3: -0.2847775, 0.2625576, -0.3186571, 0.3060418, -0.5908194, 0.5812147
4: -0.3198563, 0.3912731, -0.3753735, 0.4251533, -0.7450096, 0.7666466
5: -0.4412445, 0.5137674, -0.4906013, 0.5632069, -1.0044514, 1.0043687
6: -0.1401268, 1.2940531, -0.2510033, 1.3516467, -1.4917735, 1.5450563
7: -0.3624918, 0.4597524, -0.4133231, 0.5118459, -0.8743377, 0.8730755
8: -0.3745142, 0.4252069, -0.4205579, 0.4768797, -0.8513938, 0.8457648
9: -0.2509581, 0.2981041, -0.2897969, 0.3373972, -0.5883553, 0.5879011

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8876672, upper bound: 1.8877824
time: 2.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8876672, upper bound: 1.8883936
time: 2.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1897601, 0.8728909, -1.0782654, 2.0087161, -2.1984763, 1.9511564
1: -0.2865112, 0.3199377, -0.9632764, 0.9426547, -1.2291659, 1.2832141
2: -0.3732556, 0.3783560, -1.0389481, 1.2676729, -1.6409285, 1.4173040
3: -0.2622748, 0.2185330, -0.8072808, 0.9938859, -1.2561607, 1.0258138
4: -0.2799808, 0.3558938, -0.9951460, 1.3652675, -1.6452483, 1.3510399
5: -0.4171091, 0.4598510, -1.2644705, 1.3756353, -1.7927444, 1.7243215
6: -0.0791348, 1.2796773, -1.4329464, 1.7783635, -1.8574982, 2.7126236
7: -0.3149754, 0.4231526, -1.2031034, 1.1259530, -1.4409283, 1.6262560
8: -0.3345000, 0.3732905, -1.0825453, 1.3686163, -1.7031163, 1.4558358
9: -0.2072807, 0.2491801, -0.9547254, 1.1159703, -1.3232510, 1.2039056

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8492685, upper bound: 1.8190568
time: 2.05 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8553046, upper bound: 1.8229575
time: 2.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2312664, 0.9245848, -1.0297801, 1.9405415, -2.1718080, 1.9543650
1: -0.3200167, 0.3535223, -0.9285418, 0.9139240, -1.2339406, 1.2820641
2: -0.4043038, 0.4218536, -1.0042973, 1.2240520, -1.6283557, 1.4261508
3: -0.2848636, 0.2589081, -0.7857525, 0.9563106, -1.2411742, 1.0446606
4: -0.3194896, 0.3877837, -0.9586678, 1.3129430, -1.6324326, 1.3464515
5: -0.4435607, 0.5076071, -1.2135394, 1.3362877, -1.7798485, 1.7211465
6: -0.1434163, 1.2959421, -1.3574235, 1.7393484, -1.8827647, 2.6533656
7: -0.3590941, 0.4596763, -1.1596227, 1.0893941, -1.4484882, 1.6192989
8: -0.3708312, 0.4204896, -1.0486935, 1.3206129, -1.6914440, 1.4691832
9: -0.2469876, 0.2921115, -0.9191711, 1.0746045, -1.3215921, 1.2112826

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8622186, upper bound: 1.8290884
time: 2.25 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8622186, upper bound: 1.8300619
time: 2.14 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1924441, 0.8670551, -0.9819732, 1.8954829, -2.0879269, 1.8490283
1: -0.2868274, 0.3235715, -0.8955647, 0.8792918, -1.1661192, 1.2191361
2: -0.3723840, 0.3825782, -0.9726931, 1.1749101, -1.5472940, 1.3552713
3: -0.2623073, 0.2225588, -0.7482576, 0.9160191, -1.1783264, 0.9708164
4: -0.2806364, 0.3597653, -0.9253836, 1.2520229, -1.5326593, 1.2851489
5: -0.4149271, 0.4666748, -1.1690185, 1.2821004, -1.6970274, 1.6356933
6: -0.0768080, 1.2816662, -1.2983357, 1.7264156, -1.8032236, 2.5800018
7: -0.3188342, 0.4235124, -1.1126572, 1.0541687, -1.3730030, 1.5361696
8: -0.3386916, 0.3786046, -1.0049849, 1.2643154, -1.6030070, 1.3835895
9: -0.2115701, 0.2556143, -0.8807129, 1.0247594, -1.2363296, 1.1363273

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8455618, upper bound: 1.8184808
time: 2.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8523301, upper bound: 1.8229489
time: 1.92 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2335326, 0.9181919, -0.9211240, 1.8135158, -2.0470483, 1.8393159
1: -0.3200645, 0.3568251, -0.8520526, 0.8426654, -1.1627299, 1.2088777
2: -0.4032812, 0.4255934, -0.9294169, 1.1196961, -1.5229774, 1.3550103
3: -0.2847775, 0.2625576, -0.7144972, 0.8687490, -1.1535265, 0.9770548
4: -0.3198563, 0.3912731, -0.8798953, 1.1857889, -1.5056452, 1.2711685
5: -0.4412445, 0.5137674, -1.1057653, 1.2310206, -1.6722651, 1.6195327
6: -0.1401268, 1.2940531, -1.2055000, 1.6806054, -1.8207322, 2.4995532
7: -0.3624918, 0.4597524, -1.0578833, 1.0082471, -1.3707390, 1.5176356
8: -0.3745142, 0.4252069, -0.9613227, 1.2031422, -1.5776563, 1.3865296
9: -0.2509581, 0.2981041, -0.8361591, 0.9721631, -1.2231212, 1.1342633

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8579487, upper bound: 1.8287593
time: 2.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8579487, upper bound: 1.8300520
time: 2.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.0297801, 1.9405415, -0.2411250, 0.9525933, -1.9823735, 2.1816664
1: -0.9285418, 0.9139240, -0.3315523, 0.3590912, -1.2876329, 1.2454762
2: -1.0042973, 1.2240520, -0.4165475, 0.4303552, -1.4346524, 1.6405995
3: -0.7857525, 0.9563106, -0.2929312, 0.2667088, -1.0524614, 1.2492418
4: -0.9586678, 1.3129430, -0.3323193, 0.3925612, -1.3512290, 1.6452622
5: -1.2135394, 1.3362877, -0.4560243, 0.5131917, -1.7267311, 1.7923121
6: -1.3574235, 1.7393484, -0.1708469, 1.3109812, -2.6684046, 1.9101954
7: -1.1596227, 1.0893941, -0.3687511, 0.4714033, -1.6310260, 1.4581453
8: -1.0486935, 1.3206129, -0.3782984, 0.4283228, -1.4770163, 1.6989113
9: -0.9191711, 1.0746045, -0.2540949, 0.2968756, -1.2160467, 1.3286995

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8105656, upper bound: 1.8214155
time: 1.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8286572, upper bound: 1.8412604
time: 2.20 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.0297801, 1.9405415, -0.2080209, 0.8934053, -1.9231855, 2.1485624
1: -0.9285418, 0.9139240, -0.3006566, 0.3352100, -1.2637517, 1.2145807
2: -1.0042973, 1.2240520, -0.3862479, 0.3978687, -1.4021660, 1.6102998
3: -0.7857525, 0.9563106, -0.2719615, 0.2365722, -1.0223248, 1.2282721
4: -0.9586678, 1.3129430, -0.2967555, 0.3706242, -1.3292919, 1.6096985
5: -1.2135394, 1.3362877, -0.4279347, 0.4824034, -1.6959428, 1.7642224
6: -1.3574235, 1.7393484, -0.1057143, 1.2866631, -2.6440866, 1.8450627
7: -1.1596227, 1.0893941, -0.3345661, 0.4386751, -1.5982978, 1.4239602
8: -1.0486935, 1.3206129, -0.3514073, 0.3948846, -1.4435781, 1.6720202
9: -0.9191711, 1.0746045, -0.2249337, 0.2691265, -1.1882975, 1.2995383

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8105656, upper bound: 1.8304043
time: 2.23 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8286572, upper bound: 1.8434084
time: 1.89 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.65 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8941367, upper bound: 1.8869976
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8941368, upper bound: 1.8876707
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8956920, upper bound: 1.8879055
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8956920, upper bound: 1.8884073
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8875684, upper bound: 1.8869520
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8877206, upper bound: 1.8876575
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8876672, upper bound: 1.8877824
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8876672, upper bound: 1.8883936
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8492685, upper bound: 1.8190568
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8553046, upper bound: 1.8229575
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8622186, upper bound: 1.8290884
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8622186, upper bound: 1.8300619
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8455618, upper bound: 1.8184808
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8523301, upper bound: 1.8229489
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8579487, upper bound: 1.8287593
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8579487, upper bound: 1.8300520
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8105656, upper bound: 1.8214155
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8286572, upper bound: 1.8412604
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8105656, upper bound: 1.8304043
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.65
Output dim: 6, lower bound: -1.8286572, upper bound: 1.8434084

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1727380, 0.8818430, -0.3421358, 1.1457087, -1.3184466, 1.2239788
1: -0.2803682, 0.3021118, -0.4291425, 0.4381517, -0.7185199, 0.7312542
2: -0.3704301, 0.3572927, -0.5186012, 0.5312834, -0.9017135, 0.8758939
3: -0.2583599, 0.1974606, -0.3576490, 0.3650495, -0.6234094, 0.5551096
4: -0.2699463, 0.3396465, -0.4429360, 0.4669465, -0.7368928, 0.7825825
5: -0.4193127, 0.4340304, -0.5479226, 0.6260943, -1.0454071, 0.9819530
6: -0.0769970, 1.2923079, -0.3825551, 1.4257615, -1.5027585, 1.6748630
7: -0.2947885, 0.4143978, -0.4778590, 0.5710557, -0.8658441, 0.8922568
8: -0.3249078, 0.3497603, -0.4877410, 0.5418093, -0.8667172, 0.8375012
9: -0.1839045, 0.2226694, -0.3456177, 0.3881585, -0.5720630, 0.5682871

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8941367, upper bound: 1.8869976
time: 2.21 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8941367, upper bound: 1.8869976
time: 2.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1639524, 0.8425212, -0.3804277, 1.2064865, -1.3704388, 1.2229489
1: -0.2658233, 0.2988694, -0.4633941, 0.4745054, -0.7403287, 0.7622635
2: -0.3543407, 0.3511295, -0.5554780, 0.5716958, -0.9260365, 0.9066074
3: -0.2483708, 0.1927724, -0.3783262, 0.4129742, -0.6613449, 0.5710987
4: -0.2553498, 0.3363352, -0.4840003, 0.5124459, -0.7677957, 0.8203355
5: -0.4018022, 0.4308612, -0.5774873, 0.6789486, -1.0807508, 1.0083485
6: -0.0397051, 1.2730930, -0.4533845, 1.4478763, -1.4875815, 1.7264775
7: -0.2871540, 0.4005629, -0.5270098, 0.6050475, -0.8922015, 0.9275727
8: -0.3121681, 0.3436697, -0.5269501, 0.5902122, -0.9023802, 0.8706198
9: -0.1817149, 0.2223655, -0.3914892, 0.4306548, -0.6123697, 0.6138546

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8941368, upper bound: 1.8876707
time: 2.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8941368, upper bound: 1.8876707
time: 2.45 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.3461125, 1.1314895, -1.3697586, 1.2969071
1: -0.3297816, 0.3564569, -0.4278438, 0.4452750, -0.7750566, 0.7843007
2: -0.4150444, 0.4270655, -0.5159276, 0.5378709, -0.9529153, 0.9429931
3: -0.2916944, 0.2636827, -0.3558165, 0.3727614, -0.6644558, 0.6194992
4: -0.3300874, 0.3899961, -0.4422640, 0.4765136, -0.8066009, 0.8322600
5: -0.4548751, 0.5089113, -0.5435375, 0.6401211, -1.0949962, 1.0524487
6: -0.1680785, 1.3109281, -0.3732235, 1.4062937, -1.5743723, 1.6841516
7: -0.3655823, 0.4693472, -0.4846156, 0.5699029, -0.9354852, 0.9539628
8: -0.3765907, 0.4246833, -0.4833164, 0.5512007, -0.9277915, 0.9079997
9: -0.2510976, 0.2932430, -0.3532256, 0.3990798, -0.6501774, 0.6464686

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8941753, upper bound: 1.8879055
time: 2.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8941753, upper bound: 1.8879055
time: 2.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.3461125, 1.1314895, -1.3366139, 1.2376128
1: -0.2987980, 0.3325009, -0.4278438, 0.4452750, -0.7440730, 0.7603447
2: -0.3846641, 0.3945331, -0.5159276, 0.5378709, -0.9225350, 0.9104607
3: -0.2706087, 0.2334781, -0.3558165, 0.3727614, -0.6433702, 0.5892946
4: -0.2944325, 0.3679506, -0.4422640, 0.4765136, -0.7709461, 0.8102145
5: -0.4267558, 0.4781152, -0.5435375, 0.6401211, -1.0668769, 1.0216527
6: -0.1026776, 1.2858521, -0.3732235, 1.4062937, -1.5089715, 1.6590756
7: -0.3313026, 0.4365475, -0.4846156, 0.5699029, -0.9012054, 0.9211631
8: -0.3483621, 0.3910702, -0.4833164, 0.5512007, -0.8995628, 0.8743865
9: -0.2218433, 0.2653263, -0.3532256, 0.3990798, -0.6209230, 0.6185519

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8941753, upper bound: 1.8884073
time: 2.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8941753, upper bound: 1.8884073
time: 2.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1834476, 0.8884902, -0.2856544, 1.0479085, -1.2313561, 1.1741445
1: -0.2870093, 0.3123165, -0.3773643, 0.3903411, -0.6773504, 0.6896809
2: -0.3761815, 0.3698201, -0.4630825, 0.4733013, -0.8494828, 0.8329027
3: -0.2631493, 0.2091594, -0.3240432, 0.3049774, -0.5681267, 0.5332026
4: -0.2785701, 0.3489483, -0.3827398, 0.4225841, -0.7011542, 0.7316881
5: -0.4233587, 0.4494445, -0.5007772, 0.5567204, -0.9800791, 0.9502217
6: -0.0874342, 1.2941815, -0.2737164, 1.3764203, -1.4638544, 1.5678979
7: -0.3066084, 0.4223768, -0.4137304, 0.5186859, -0.8252943, 0.8361071
8: -0.3318501, 0.3637381, -0.4318082, 0.4745359, -0.8063860, 0.7955463
9: -0.1955644, 0.2358899, -0.2879019, 0.3316465, -0.5272110, 0.5237918

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8870008, upper bound: 1.8868402
time: 3.26 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8870034, upper bound: 1.8862572
time: 2.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1662713, 0.8360838, -0.3234297, 1.1010146, -1.2672859, 1.1595135
1: -0.2658493, 0.3025315, -0.4087837, 0.4233904, -0.6892396, 0.7113152
2: -0.3531312, 0.3553646, -0.4957850, 0.5131722, -0.8663034, 0.8511496
3: -0.2482097, 0.1968333, -0.3444104, 0.3452059, -0.5934156, 0.5412437
4: -0.2559156, 0.3397877, -0.4199743, 0.4530167, -0.7089322, 0.7597620
5: -0.3993143, 0.4370752, -0.5277154, 0.6054564, -1.0047706, 0.9647906
6: -0.0371284, 1.2764713, -0.3362368, 1.3972590, -1.4343874, 1.6127081
7: -0.2907535, 0.4007893, -0.4565795, 0.5513378, -0.8420913, 0.8573688
8: -0.3163596, 0.3490134, -0.4642868, 0.5206126, -0.8369722, 0.8133001
9: -0.1860418, 0.2282734, -0.3281225, 0.3720302, -0.5580720, 0.5563959

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8870455, upper bound: 1.8873423
time: 2.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8870455, upper bound: 1.8876575
time: 2.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -0.2839092, 1.0218475, -1.2784026, 1.2583494
1: -0.3448424, 0.3715456, -0.3700308, 0.3923557, -0.7371981, 0.7415764
2: -0.4289511, 0.4466715, -0.4536291, 0.4743348, -0.9032859, 0.9003006
3: -0.3015637, 0.2813208, -0.3186571, 0.3060418, -0.6076056, 0.5999779
4: -0.3472072, 0.4048828, -0.3753735, 0.4251533, -0.7723604, 0.7802563
5: -0.4669372, 0.5324135, -0.4906013, 0.5632069, -1.0301441, 1.0230148
6: -0.1975362, 1.3229021, -0.2510033, 1.3516467, -1.5491829, 1.5739053
7: -0.3851250, 0.4853222, -0.4133231, 0.5118459, -0.8969709, 0.8986453
8: -0.3923242, 0.4465339, -0.4205579, 0.4768797, -0.8692039, 0.8670918
9: -0.2674738, 0.3125915, -0.2897969, 0.3373972, -0.6048710, 0.6023885

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8866438, upper bound: 1.8875684
time: 2.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8873578, upper bound: 1.8877206
time: 3.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2067769, 0.8844675, -0.2839092, 1.0218475, -1.2286243, 1.1683767
1: -0.2982824, 0.3353040, -0.3700308, 0.3923557, -0.6906381, 0.7053348
2: -0.3830329, 0.3976636, -0.4536291, 0.4743348, -0.8573677, 0.8512927
3: -0.2701186, 0.2365419, -0.3186571, 0.3060418, -0.5761604, 0.5551989
4: -0.2941514, 0.3709590, -0.3753735, 0.4251533, -0.7193047, 0.7463325
5: -0.4239522, 0.4835648, -0.4906013, 0.5632069, -0.9871591, 0.9741660
6: -0.0985513, 1.2864097, -0.2510033, 1.3516467, -1.4501979, 1.5374130
7: -0.3340706, 0.4360552, -0.4133231, 0.5118459, -0.8459166, 0.8493783
8: -0.3514939, 0.3950947, -0.4205579, 0.4768797, -0.8283736, 0.8156526
9: -0.2252686, 0.2706831, -0.2897969, 0.3373972, -0.5626657, 0.5604800

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8866438, upper bound: 1.8882853
time: 2.55 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8873578, upper bound: 1.8883800
time: 3.88 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1727380, 0.8818430, -1.0110528, 1.9387521, -2.1114900, 1.8928958
1: -0.2803682, 0.3021118, -0.9166154, 0.8965663, -1.1769345, 1.2187271
2: -0.3704301, 0.3572927, -0.9940382, 1.2008195, -1.5712496, 1.3513309
3: -0.2583599, 0.1974606, -0.7650340, 0.9381437, -1.1965036, 0.9624947
4: -0.2699463, 0.3396465, -0.9474397, 1.2826151, -1.5525614, 1.2870862
5: -0.4193127, 0.4340304, -1.1995890, 1.3062242, -1.7255368, 1.6336194
6: -0.0769970, 1.2923079, -1.3449259, 1.7533456, -1.8303427, 2.6372337
7: -0.2947885, 0.4143978, -1.1382580, 1.0765336, -1.3713220, 1.5526558
8: -0.3249078, 0.3497603, -1.0258131, 1.2933402, -1.6182480, 1.3755734
9: -0.1839045, 0.2226694, -0.9010510, 1.0490751, -1.2329797, 1.1237204

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 122

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8481401, upper bound: 1.8188808
time: 2.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8487750, upper bound: 1.8188808
time: 2.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1639524, 0.8425212, -1.0782654, 2.0087161, -2.1726685, 1.9207866
1: -0.2658233, 0.2988694, -0.9632764, 0.9426547, -1.2084780, 1.2621458
2: -0.3543407, 0.3511295, -1.0389481, 1.2676729, -1.6220136, 1.3900776
3: -0.2483708, 0.1927724, -0.8072808, 0.9938859, -1.2422566, 1.0000532
4: -0.2553498, 0.3363352, -0.9951460, 1.3652675, -1.6206173, 1.3314812
5: -0.4018022, 0.4308612, -1.2644705, 1.3756353, -1.7774374, 1.6953316
6: -0.0397051, 1.2730930, -1.4329464, 1.7783635, -1.8180685, 2.7060394
7: -0.2871540, 0.4005629, -1.2031034, 1.1259530, -1.4131070, 1.6036663
8: -0.3121681, 0.3436697, -1.0825453, 1.3686163, -1.6807845, 1.4262149
9: -0.1817149, 0.2223655, -0.9547254, 1.1159703, -1.2976851, 1.1770909

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 122

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8546660, upper bound: 1.8229465
time: 2.15 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8548543, upper bound: 1.8229270
time: 2.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -1.0297801, 1.9405415, -2.1788106, 1.9805747
1: -0.3297816, 0.3564569, -0.9285418, 0.9139240, -1.2437056, 1.2849987
2: -0.4150444, 0.4270655, -1.0042973, 1.2240520, -1.6390963, 1.4313627
3: -0.2916944, 0.2636827, -0.7857525, 0.9563106, -1.2480049, 1.0494353
4: -0.3300874, 0.3899961, -0.9586678, 1.3129430, -1.6430304, 1.3486638
5: -0.4548751, 0.5089113, -1.2135394, 1.3362877, -1.7911627, 1.7224506
6: -0.1680785, 1.3109281, -1.3574235, 1.7393484, -1.9074270, 2.6683517
7: -0.3655823, 0.4693472, -1.1596227, 1.0893941, -1.4549764, 1.6289699
8: -0.3765907, 0.4246833, -1.0486935, 1.3206129, -1.6972036, 1.4733769
9: -0.2510976, 0.2932430, -0.9191711, 1.0746045, -1.3257021, 1.2124140

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 122

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8249357, upper bound: 1.8122282
time: 2.18 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8553046, upper bound: 1.8290884
time: 2.52 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -1.0297801, 1.9405415, -2.1456659, 1.9212805
1: -0.2987980, 0.3325009, -0.9285418, 0.9139240, -1.2127221, 1.2610426
2: -0.3846641, 0.3945331, -1.0042973, 1.2240520, -1.6087160, 1.3988304
3: -0.2706087, 0.2334781, -0.7857525, 0.9563106, -1.2269193, 1.0192306
4: -0.2944325, 0.3679506, -0.9586678, 1.3129430, -1.6073755, 1.3266184
5: -0.4267558, 0.4781152, -1.2135394, 1.3362877, -1.7630435, 1.6916546
6: -0.1026776, 1.2858521, -1.3574235, 1.7393484, -1.8420260, 2.6432757
7: -0.3313026, 0.4365475, -1.1596227, 1.0893941, -1.4206966, 1.5961702
8: -0.3483621, 0.3910702, -1.0486935, 1.3206129, -1.6689750, 1.4397638
9: -0.2218433, 0.2653263, -0.9191711, 1.0746045, -1.2964478, 1.1844974

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 122

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8249357, upper bound: 1.8164043
time: 2.22 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8553046, upper bound: 1.8300619
time: 2.30 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1834476, 0.8884902, -0.9149100, 1.8256900, -2.0091376, 1.8034002
1: -0.2870093, 0.3123165, -0.8489914, 0.8332641, -1.1202734, 1.1613079
2: -0.3761815, 0.3698201, -0.9278882, 1.1081617, -1.4843432, 1.2977083
3: -0.2631493, 0.2091594, -0.7070364, 0.8603744, -1.1235237, 0.9161958
4: -0.2785701, 0.3489483, -0.8777792, 1.1694527, -1.4480228, 1.2267276
5: -0.4233587, 0.4494445, -1.1042717, 1.2128201, -1.6361787, 1.5537162
6: -0.0874342, 1.2941815, -1.2105746, 1.7015208, -1.7889550, 2.5047560
7: -0.3066084, 0.4223768, -1.0479295, 1.0049019, -1.3115103, 1.4703063
8: -0.3318501, 0.3637381, -0.9495711, 1.1891594, -1.5210094, 1.3133092
9: -0.1955644, 0.2358899, -0.8270821, 0.9579507, -1.1535151, 1.0629721

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8448715, upper bound: 1.8183359
time: 2.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454011, upper bound: 1.8183320
time: 2.28 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1662713, 0.8360838, -0.9819732, 1.8954829, -2.0617542, 1.8180571
1: -0.2658493, 0.3025315, -0.8955647, 0.8792918, -1.1451410, 1.1980962
2: -0.3531312, 0.3553646, -0.9726931, 1.1749101, -1.5280412, 1.3280576
3: -0.2482097, 0.1968333, -0.7482576, 0.9160191, -1.1642288, 0.9450909
4: -0.2559156, 0.3397877, -0.9253836, 1.2520229, -1.5079384, 1.2651713
5: -0.3993143, 0.4370752, -1.1690185, 1.2821004, -1.6814147, 1.6060936
6: -0.0371284, 1.2764713, -1.2983357, 1.7264156, -1.7635441, 2.5748069
7: -0.2907535, 0.4007893, -1.1126572, 1.0541687, -1.3449222, 1.5134465
8: -0.3163596, 0.3490134, -1.0049849, 1.2643154, -1.5806750, 1.3539982
9: -0.1860418, 0.2282734, -0.8807129, 1.0247594, -1.2108011, 1.1089863

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8518163, upper bound: 1.8229442
time: 2.00 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8522239, upper bound: 1.8229260
time: 2.28 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -0.9211240, 1.8135158, -2.0700710, 1.8955643
1: -0.3448424, 0.3715456, -0.8520526, 0.8426654, -1.1875079, 1.2235981
2: -0.4289511, 0.4466715, -0.9294169, 1.1196961, -1.5486473, 1.3760884
3: -0.3015637, 0.2813208, -0.7144972, 0.8687490, -1.1703126, 0.9958180
4: -0.3472072, 0.4048828, -0.8798953, 1.1857889, -1.5329961, 1.2847780
5: -0.4669372, 0.5324135, -1.1057653, 1.2310206, -1.6979578, 1.6381788
6: -0.1975362, 1.3229021, -1.2055000, 1.6806054, -1.8781416, 2.5284021
7: -0.3851250, 0.4853222, -1.0578833, 1.0082471, -1.3933721, 1.5432055
8: -0.3923242, 0.4465339, -0.9613227, 1.2031422, -1.5954664, 1.4078565
9: -0.2674738, 0.3125915, -0.8361591, 0.9721631, -1.2396369, 1.1487507

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 168

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8226954, upper bound: 1.8108261
time: 2.27 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8523301, upper bound: 1.8287593
time: 2.22 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2067769, 0.8844675, -0.9211240, 1.8135158, -2.0202928, 1.8055916
1: -0.2982824, 0.3353040, -0.8520526, 0.8426654, -1.1409478, 1.1873565
2: -0.3830329, 0.3976636, -0.9294169, 1.1196961, -1.5027291, 1.3270805
3: -0.2701186, 0.2365419, -0.7144972, 0.8687490, -1.1388676, 0.9510391
4: -0.2941514, 0.3709590, -0.8798953, 1.1857889, -1.4799402, 1.2508543
5: -0.4239522, 0.4835648, -1.1057653, 1.2310206, -1.6549728, 1.5893302
6: -0.0985513, 1.2864097, -1.2055000, 1.6806054, -1.7791567, 2.4919097
7: -0.3340706, 0.4360552, -1.0578833, 1.0082471, -1.3423178, 1.4939384
8: -0.3514939, 0.3950947, -0.9613227, 1.2031422, -1.5546360, 1.3564173
9: -0.2252686, 0.2706831, -0.8361591, 0.9721631, -1.1974317, 1.1068422

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 168

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8226954, upper bound: 1.8163913
time: 2.43 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8523301, upper bound: 1.8300520
time: 2.83 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.9833275, 1.8930256, -0.2411250, 0.9525933, -1.9359208, 2.1341505
1: -0.8961952, 0.8818668, -0.3315523, 0.3590912, -1.2552863, 1.2134192
2: -0.9733252, 1.1776061, -0.4165475, 0.4303552, -1.4036803, 1.5941536
3: -0.7523167, 0.9175037, -0.2929312, 0.2667088, -1.0190256, 1.2104349
4: -0.9256110, 1.2548698, -0.3323193, 0.3925612, -1.3181722, 1.5871892
5: -1.1687191, 1.2888355, -0.4560243, 0.5131917, -1.6819108, 1.7448599
6: -1.2971220, 1.7235489, -0.1708469, 1.3109812, -2.6081033, 1.8943958
7: -1.1144310, 1.0552229, -0.3687511, 0.4714033, -1.5858343, 1.4239740
8: -1.0099509, 1.2683618, -0.3782984, 0.4283228, -1.4382737, 1.6466602
9: -0.8812496, 1.0278685, -0.2540949, 0.2968756, -1.1781253, 1.2819635

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 189

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8286572, upper bound: 1.8412604
time: 1.99 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8286572, upper bound: 1.8412604
time: 2.10 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.9833275, 1.8930256, -0.2080209, 0.8934053, -1.8767328, 2.1010466
1: -0.8961952, 0.8818668, -0.3006566, 0.3352100, -1.2314053, 1.1825235
2: -0.9733252, 1.1776061, -0.3862479, 0.3978687, -1.3711939, 1.5638540
3: -0.7523167, 0.9175037, -0.2719615, 0.2365722, -0.9888889, 1.1894653
4: -0.9256110, 1.2548698, -0.2967555, 0.3706242, -1.2962352, 1.5516253
5: -1.1687191, 1.2888355, -0.4279347, 0.4824034, -1.6511225, 1.7167702
6: -1.2971220, 1.7235489, -0.1057143, 1.2866631, -2.5837851, 1.8292632
7: -1.1144310, 1.0552229, -0.3345661, 0.4386751, -1.5531061, 1.3897890
8: -1.0099509, 1.2683618, -0.3514073, 0.3948846, -1.4048355, 1.6197691
9: -0.8812496, 1.0278685, -0.2249337, 0.2691265, -1.1503761, 1.2528023

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 189

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8297394, upper bound: 1.8434084
time: 2.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8297394, upper bound: 1.8434084
time: 2.02 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.92 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8941367, upper bound: 1.8869976
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8941367, upper bound: 1.8869976
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8941368, upper bound: 1.8876707
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8941368, upper bound: 1.8876707
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8941753, upper bound: 1.8879055
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8941753, upper bound: 1.8879055
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8941753, upper bound: 1.8884073
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8941753, upper bound: 1.8884073
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8870008, upper bound: 1.8868402
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8870034, upper bound: 1.8862572
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8870455, upper bound: 1.8873423
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8870455, upper bound: 1.8876575
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8866438, upper bound: 1.8875684
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8873578, upper bound: 1.8877206
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8866438, upper bound: 1.8882853
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8873578, upper bound: 1.8883800
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8481401, upper bound: 1.8188808
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8487750, upper bound: 1.8188808
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8546660, upper bound: 1.8229465
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8548543, upper bound: 1.8229270
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8249357, upper bound: 1.8122282
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8553046, upper bound: 1.8290884
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8249357, upper bound: 1.8164043
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8553046, upper bound: 1.8300619
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8448715, upper bound: 1.8183359
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8454011, upper bound: 1.8183320
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8518163, upper bound: 1.8229442
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8522239, upper bound: 1.8229260
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8226954, upper bound: 1.8108261
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8523301, upper bound: 1.8287593
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8226954, upper bound: 1.8163913
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8523301, upper bound: 1.8300520
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8286572, upper bound: 1.8412604
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8286572, upper bound: 1.8412604
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8297394, upper bound: 1.8434084
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.92
Output dim: 6, lower bound: -1.8297394, upper bound: 1.8434084

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1727380, 0.8818430, -0.2007700, 0.9052042, -1.0779421, 1.0826130
1: -0.2803682, 0.3021118, -0.2998521, 0.3260188, -0.6063870, 0.6019639
2: -0.3704301, 0.3572927, -0.3873505, 0.3875303, -0.7579604, 0.7446432
3: -0.2583599, 0.1974606, -0.2715769, 0.2269373, -0.4852972, 0.4690376
4: -0.2699463, 0.3396465, -0.2946766, 0.3612121, -0.6311585, 0.6343231
5: -0.4193127, 0.4340304, -0.4315962, 0.4654199, -0.8847325, 0.8656266
6: -0.0769970, 1.2923079, -0.1107771, 1.2931693, -1.3701663, 1.4030850
7: -0.2947885, 0.4143978, -0.3255377, 0.4367231, -0.7315115, 0.7399356
8: -0.3249078, 0.3497603, -0.3449024, 0.3821782, -0.7070861, 0.6946627
9: -0.1839045, 0.2226694, -0.2148761, 0.2542391, -0.4381436, 0.4375455

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8937388, upper bound: 1.8863091
time: 2.27 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8931445, upper bound: 1.8863106
time: 2.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1727380, 0.8818430, -0.2200130, 0.9290566, -1.1017946, 1.1018560
1: -0.2803682, 0.3021118, -0.3156103, 0.3417507, -0.6221189, 0.6177220
2: -0.3704301, 0.3572927, -0.4019283, 0.4081444, -0.7785745, 0.7592210
3: -0.2583599, 0.1974606, -0.2819648, 0.2455107, -0.5038706, 0.4794255
4: -0.2699463, 0.3396465, -0.3126071, 0.3765858, -0.6465321, 0.6522536
5: -0.4193127, 0.4340304, -0.4437953, 0.4899528, -0.9092655, 0.8778257
6: -0.0769970, 1.2923079, -0.1415467, 1.3052105, -1.3822075, 1.4338546
7: -0.2947885, 0.4143978, -0.3460531, 0.4534410, -0.7482294, 0.7604509
8: -0.3249078, 0.3497603, -0.3614716, 0.4047639, -0.7296717, 0.7112318
9: -0.1839045, 0.2226694, -0.2321969, 0.2743294, -0.4582339, 0.4548663

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8937388, upper bound: 1.8863091
time: 2.47 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8931445, upper bound: 1.8863106
time: 2.73 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1639524, 0.8425212, -0.2382691, 0.9507946, -1.1147469, 1.0807903
1: -0.2658233, 0.2988694, -0.3297816, 0.3564569, -0.6222802, 0.6286510
2: -0.3543407, 0.3511295, -0.4150444, 0.4270655, -0.7814062, 0.7661738
3: -0.2483708, 0.1927724, -0.2916944, 0.2636827, -0.5120535, 0.4844669
4: -0.2553498, 0.3363352, -0.3300874, 0.3899961, -0.6453458, 0.6664225
5: -0.4018022, 0.4308612, -0.4548751, 0.5089113, -0.9107134, 0.8857362
6: -0.0397051, 1.2730930, -0.1680785, 1.3109281, -1.3506331, 1.4411715
7: -0.2871540, 0.4005629, -0.3655823, 0.4693472, -0.7565012, 0.7661452
8: -0.3121681, 0.3436697, -0.3765907, 0.4246833, -0.7368515, 0.7202604
9: -0.1817149, 0.2223655, -0.2510976, 0.2932430, -0.4749579, 0.4734630

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8937436, upper bound: 1.8871205
time: 2.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8931764, upper bound: 1.8871236
time: 3.04 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1639524, 0.8425212, -0.2566187, 0.9745614, -1.1385138, 1.0991399
1: -0.2658233, 0.2988694, -0.3449113, 0.3714673, -0.6372906, 0.6437807
2: -0.3543407, 0.3511295, -0.4290012, 0.4467482, -0.8010889, 0.7801306
3: -0.2483708, 0.1927724, -0.3016051, 0.2813783, -0.5297490, 0.4943776
4: -0.2553498, 0.3363352, -0.3472668, 0.4047099, -0.6600597, 0.6836020
5: -0.4018022, 0.4308612, -0.4669960, 0.5325068, -0.9343090, 0.8978571
6: -0.0397051, 1.2730930, -0.1977043, 1.3230022, -1.3627074, 1.4707973
7: -0.2871540, 0.4005629, -0.3851508, 0.4853825, -0.7725365, 0.7857137
8: -0.3121681, 0.3436697, -0.3923964, 0.4462870, -0.7584552, 0.7360661
9: -0.1817149, 0.2223655, -0.2675011, 0.3124065, -0.4941214, 0.4898666

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8937436, upper bound: 1.8871205
time: 2.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8931764, upper bound: 1.8871235
time: 2.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.2051244, 0.8915004, -1.1297694, 1.1559190
1: -0.3297816, 0.3564569, -0.2987980, 0.3325009, -0.6622825, 0.6552550
2: -0.4150444, 0.4270655, -0.3846641, 0.3945331, -0.8095775, 0.8117296
3: -0.2916944, 0.2636827, -0.2706087, 0.2334781, -0.5251725, 0.5342914
4: -0.3300874, 0.3899961, -0.2944325, 0.3679506, -0.6980379, 0.6844286
5: -0.4548751, 0.5089113, -0.4267558, 0.4781152, -0.9329903, 0.9356670
6: -0.1680785, 1.3109281, -0.1026776, 1.2858521, -1.4539306, 1.4136057
7: -0.3655823, 0.4693472, -0.3313026, 0.4365475, -0.8021299, 0.8006498
8: -0.3765907, 0.4246833, -0.3483621, 0.3910702, -0.7676610, 0.7730454
9: -0.2510976, 0.2932430, -0.2218433, 0.2653263, -0.5164238, 0.5150863

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8956502, upper bound: 1.8872369
time: 2.49 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8956502, upper bound: 1.8878554
time: 3.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.2067769, 0.8844675, -1.1227366, 1.1575714
1: -0.3297816, 0.3564569, -0.2982824, 0.3353040, -0.6650856, 0.6547394
2: -0.4150444, 0.4270655, -0.3830329, 0.3976636, -0.8127080, 0.8100984
3: -0.2916944, 0.2636827, -0.2701186, 0.2365419, -0.5282363, 0.5338013
4: -0.3300874, 0.3899961, -0.2941514, 0.3709590, -0.7010463, 0.6841474
5: -0.4548751, 0.5089113, -0.4239522, 0.4835648, -0.9384398, 0.9328635
6: -0.1680785, 1.3109281, -0.0985513, 1.2864097, -1.4544883, 1.4094794
7: -0.3655823, 0.4693472, -0.3340706, 0.4360552, -0.8016376, 0.8034178
8: -0.3765907, 0.4246833, -0.3514939, 0.3950947, -0.7716854, 0.7761772
9: -0.2510976, 0.2932430, -0.2252686, 0.2706831, -0.5217806, 0.5185115

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8956502, upper bound: 1.8872369
time: 3.04 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8956502, upper bound: 1.8878554
time: 2.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.2051244, 0.8915004, -1.0966247, 1.0966247
1: -0.2987980, 0.3325009, -0.2987980, 0.3325009, -0.6312989, 0.6312989
2: -0.3846641, 0.3945331, -0.3846641, 0.3945331, -0.7791972, 0.7791972
3: -0.2706087, 0.2334781, -0.2706087, 0.2334781, -0.5040869, 0.5040869
4: -0.2944325, 0.3679506, -0.2944325, 0.3679506, -0.6623831, 0.6623831
5: -0.4267558, 0.4781152, -0.4267558, 0.4781152, -0.9048710, 0.9048710
6: -0.1026776, 1.2858521, -0.1026776, 1.2858521, -1.3885298, 1.3885298
7: -0.3313026, 0.4365475, -0.3313026, 0.4365475, -0.7678500, 0.7678500
8: -0.3483621, 0.3910702, -0.3483621, 0.3910702, -0.7394323, 0.7394323
9: -0.2218433, 0.2653263, -0.2218433, 0.2653263, -0.4871695, 0.4871695

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8945125, upper bound: 1.8878040
time: 2.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8945125, upper bound: 1.8883937
time: 2.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.2067769, 0.8844675, -1.0895919, 1.0982772
1: -0.2987980, 0.3325009, -0.2982824, 0.3353040, -0.6341020, 0.6307833
2: -0.3846641, 0.3945331, -0.3830329, 0.3976636, -0.7823277, 0.7775661
3: -0.2706087, 0.2334781, -0.2701186, 0.2365419, -0.5071506, 0.5035967
4: -0.2944325, 0.3679506, -0.2941514, 0.3709590, -0.6653916, 0.6621019
5: -0.4267558, 0.4781152, -0.4239522, 0.4835648, -0.9103206, 0.9020674
6: -0.1026776, 1.2858521, -0.0985513, 1.2864097, -1.3890874, 1.3844035
7: -0.3313026, 0.4365475, -0.3340706, 0.4360552, -0.7673578, 0.7706181
8: -0.3483621, 0.3910702, -0.3514939, 0.3950947, -0.7434567, 0.7425641
9: -0.2218433, 0.2653263, -0.2252686, 0.2706831, -0.4925264, 0.4905948

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8945125, upper bound: 1.8878040
time: 2.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8945125, upper bound: 1.8883937
time: 2.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1617692, 0.8587554, -0.1538141, 0.8618466, -1.0236158, 1.0125695
1: -0.2688225, 0.2940106, -0.2658817, 0.2849665, -0.5537890, 0.5598923
2: -0.3592864, 0.3467408, -0.3576462, 0.3373343, -0.6966207, 0.7043871
3: -0.2512638, 0.1853703, -0.2501692, 0.1720283, -0.4232920, 0.4355395
4: -0.2580197, 0.3329864, -0.2554307, 0.3245920, -0.5826116, 0.5884171
5: -0.4089479, 0.4260335, -0.4091378, 0.4124947, -0.8214425, 0.8351713
6: -0.0503951, 1.2839557, -0.0484803, 1.2887865, -1.3391817, 1.3324361
7: -0.2831461, 0.4025550, -0.2730781, 0.3999347, -0.6830808, 0.6756331
8: -0.3127802, 0.3362087, -0.3125731, 0.3208668, -0.6336470, 0.6487818
9: -0.1734019, 0.2178102, -0.1619418, 0.2112200, -0.3846219, 0.3797520

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8732932, upper bound: 1.8754558
time: 2.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8759964, upper bound: 1.8757412
time: 2.30 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1683012, 0.8686923, -0.2472226, 0.9981180, -1.1664193, 1.1159148
1: -0.2746186, 0.2994523, -0.3463250, 0.3595352, -0.6341538, 0.6457773
2: -0.3647394, 0.3532646, -0.4342042, 0.4331878, -0.7979273, 0.7874688
3: -0.2546288, 0.1934465, -0.3032127, 0.2676018, -0.5222306, 0.4966592
4: -0.2637104, 0.3379016, -0.3460106, 0.3936050, -0.6573155, 0.6839122
5: -0.4137072, 0.4329888, -0.4755965, 0.5135888, -0.9272960, 0.9085852
6: -0.0625393, 1.2873634, -0.2133097, 1.3538803, -1.4164196, 1.5006731
7: -0.2903322, 0.4082829, -0.3729191, 0.4849544, -0.7752866, 0.7812020
8: -0.3182665, 0.3455703, -0.3987117, 0.4315317, -0.7497982, 0.7442820
9: -0.1801065, 0.2216410, -0.2509396, 0.2924643, -0.4725708, 0.4725806

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8732937, upper bound: 1.8745655
time: 2.31 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8759996, upper bound: 1.8747944
time: 2.15 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1662713, 0.8360838, -0.3144604, 1.1209884, -1.2872597, 1.1505442
1: -0.2658493, 0.3025315, -0.4099888, 0.4101070, -0.6759562, 0.7125202
2: -0.3531312, 0.3553646, -0.4980230, 0.5004569, -0.8535881, 0.8533875
3: -0.2482097, 0.1968333, -0.3466946, 0.3292175, -0.5774271, 0.5435280
4: -0.2559156, 0.3397877, -0.4182196, 0.4413324, -0.6972480, 0.7580073
5: -0.3993143, 0.4370752, -0.5344468, 0.5839525, -0.9832668, 0.9715220
6: -0.0371284, 1.2764713, -0.3501613, 1.4334911, -1.4706196, 1.6266326
7: -0.2907535, 0.4007893, -0.4431561, 0.5517006, -0.8424541, 0.8439454
8: -0.3163596, 0.3490134, -0.4719550, 0.5040433, -0.8204030, 0.8209684
9: -0.1860418, 0.2282734, -0.3094426, 0.3524070, -0.5384488, 0.5377160

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8868715, upper bound: 1.8865594
time: 2.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8865121, upper bound: 1.8865610
time: 2.53 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1662713, 0.8360838, -0.2984414, 1.0645648, -1.2308362, 1.1345251
1: -0.2658493, 0.3025315, -0.3876220, 0.4007919, -0.6666411, 0.6901535
2: -0.3531312, 0.3553646, -0.4726198, 0.4868918, -0.8400229, 0.8279843
3: -0.2482097, 0.1968333, -0.3310031, 0.3174252, -0.5656348, 0.5278364
4: -0.2559156, 0.3397877, -0.3947391, 0.4326347, -0.6885502, 0.7345268
5: -0.3993143, 0.4370752, -0.5097136, 0.5723346, -0.9716489, 0.9467888
6: -0.0371284, 1.2764713, -0.2936292, 1.3843944, -1.4215229, 1.5701004
7: -0.2907535, 0.4007893, -0.4273822, 0.5297623, -0.8205158, 0.8281715
8: -0.3163596, 0.3490134, -0.4427789, 0.4892639, -0.8056235, 0.7917923
9: -0.1860418, 0.2282734, -0.2998447, 0.3449753, -0.5310171, 0.5281181

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8868715, upper bound: 1.8871052
time: 2.27 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8865121, upper bound: 1.8871076
time: 2.29 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2200975, 0.9293086, -0.2692408, 1.0338130, -1.2539105, 1.1985494
1: -0.3157351, 0.3418151, -0.3659123, 0.3767951, -0.6925302, 0.7077274
2: -0.4020479, 0.4082354, -0.4533503, 0.4559991, -0.8580470, 0.8615857
3: -0.2820313, 0.2455828, -0.3166661, 0.2880602, -0.5700915, 0.5622488
4: -0.3127112, 0.3766707, -0.3681426, 0.4104189, -0.7231301, 0.7448133
5: -0.4438757, 0.4900700, -0.4930621, 0.5398947, -0.9837704, 0.9831321
6: -0.1418602, 1.3053989, -0.2540263, 1.3758644, -1.5177245, 1.5594252
7: -0.3461611, 0.4535449, -0.3960058, 0.5056229, -0.8517839, 0.8495507
8: -0.3615938, 0.4048975, -0.4205560, 0.4564394, -0.8180332, 0.8254535
9: -0.2322302, 0.2744126, -0.2696044, 0.3135830, -0.5458131, 0.5440170

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8868402, upper bound: 1.8870008
time: 2.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8862572, upper bound: 1.8870034
time: 3.45 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -0.2578405, 0.9902911, -1.2468462, 1.2322807
1: -0.3448424, 0.3715456, -0.3493063, 0.3712159, -0.7160583, 0.7208519
2: -0.4289511, 0.4466715, -0.4346911, 0.4469483, -0.8758994, 0.8813626
3: -0.3015637, 0.2813208, -0.3048547, 0.2803797, -0.5819435, 0.5861756
4: -0.3472072, 0.4048828, -0.3506503, 0.4052950, -0.7525021, 0.7555330
5: -0.4669372, 0.5324135, -0.4738116, 0.5337979, -1.0007350, 1.0062251
6: -0.1975362, 1.3229021, -0.2117815, 1.3394281, -1.5369642, 1.5346836
7: -0.3851250, 0.4853222, -0.3854574, 0.4892491, -0.8743740, 0.8707796
8: -0.3923242, 0.4465339, -0.3988919, 0.4474792, -0.8398035, 0.8454257
9: -0.2674738, 0.3125915, -0.2641335, 0.3101937, -0.5776675, 0.5767250

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8873423, upper bound: 1.8870456
time: 2.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8873423, upper bound: 1.8870456
time: 2.29 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1694376, 0.8388355, -0.2692408, 1.0338130, -1.2032506, 1.1080763
1: -0.2681902, 0.3050637, -0.3659123, 0.3767951, -0.6449853, 0.6709760
2: -0.3551775, 0.3586025, -0.4533503, 0.4559991, -0.8111765, 0.8119528
3: -0.2497055, 0.1999865, -0.3166661, 0.2880602, -0.5377657, 0.5166526
4: -0.2588218, 0.3420937, -0.3681426, 0.4104189, -0.6692408, 0.7102363
5: -0.4008318, 0.4402800, -0.4930621, 0.5398947, -0.9407265, 0.9333421
6: -0.0411611, 1.2761184, -0.2540263, 1.3758644, -1.4170254, 1.5301447
7: -0.2940759, 0.4034359, -0.3960058, 0.5056229, -0.7996987, 0.7994418
8: -0.3186762, 0.3524678, -0.4205560, 0.4564394, -0.7751156, 0.7730238
9: -0.1893898, 0.2316148, -0.2696044, 0.3135830, -0.5029727, 0.5012192

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8874072, upper bound: 1.8877897
time: 2.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8871057, upper bound: 1.8877924
time: 2.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2067769, 0.8844675, -0.2578405, 0.9902911, -1.1970680, 1.1423080
1: -0.2982824, 0.3353040, -0.3493063, 0.3712159, -0.6694983, 0.6846103
2: -0.3830329, 0.3976636, -0.4346911, 0.4469483, -0.8299812, 0.8323547
3: -0.2701186, 0.2365419, -0.3048547, 0.2803797, -0.5504984, 0.5413966
4: -0.2941514, 0.3709590, -0.3506503, 0.4052950, -0.6994463, 0.7216092
5: -0.4239522, 0.4835648, -0.4738116, 0.5337979, -0.9577501, 0.9573764
6: -0.0985513, 1.2864097, -0.2117815, 1.3394281, -1.4379795, 1.4981912
7: -0.3340706, 0.4360552, -0.3854574, 0.4892491, -0.8233197, 0.8215126
8: -0.3514939, 0.3950947, -0.3988919, 0.4474792, -0.7989731, 0.7939866
9: -0.2252686, 0.2706831, -0.2641335, 0.3101937, -0.5354623, 0.5348166

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8880674, upper bound: 1.8877609
time: 3.20 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8880674, upper bound: 1.8883800
time: 3.39 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1532322, 0.8528119, -0.7455895, 1.6458535, -1.7990857, 1.5984014
1: -0.2631414, 0.2856838, -0.7320441, 0.7180434, -0.9811848, 1.0177279
2: -0.3543549, 0.3374994, -0.8150766, 0.9406457, -1.2950006, 1.1525760
3: -0.2481987, 0.1734768, -0.6044180, 0.7202347, -0.9684334, 0.7778947
4: -0.2526385, 0.3250853, -0.7576795, 0.9613730, -1.2140114, 1.0827649
5: -0.4053289, 0.4133819, -0.9402479, 1.0424182, -1.4477472, 1.3536298
6: -0.0409845, 1.2817119, -0.9867525, 1.6351266, -1.6761110, 2.2684646
7: -0.2733481, 0.3970807, -0.8849033, 0.8814704, -1.1548185, 1.2819841
8: -0.3083943, 0.3219871, -0.8223755, 1.0009406, -1.3093349, 1.1443627
9: -0.1636618, 0.2108611, -0.6910452, 0.7905286, -0.9541904, 0.9019064

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8322606, upper bound: 1.8046867
time: 2.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8360871, upper bound: 1.8065103
time: 2.30 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1599368, 0.8629891, -0.9339352, 1.8569810, -2.0169177, 1.7969244
1: -0.2691875, 0.2911630, -0.8631197, 0.8439838, -1.1131713, 1.1542827
2: -0.3600231, 0.3440618, -0.9425864, 1.1244006, -1.4844238, 1.2866482
3: -0.2516368, 0.1818415, -0.7181682, 0.8742954, -1.1259321, 0.9000097
4: -0.2584179, 0.3302004, -0.8926174, 1.1877359, -1.4461539, 1.2228179
5: -0.4102079, 0.4205729, -1.1249762, 1.2283423, -1.6385503, 1.5455492
6: -0.0535964, 1.2854966, -1.2431333, 1.7230086, -1.7766051, 2.5286298
7: -0.2807996, 0.4028133, -1.0639689, 1.0201579, -1.3009574, 1.4667821
8: -0.3140454, 0.3317145, -0.9665742, 1.2074312, -1.5214765, 1.2982886
9: -0.1704213, 0.2147656, -0.8390821, 0.9727294, -1.1431507, 1.0538478

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8328922, upper bound: 1.8046606
time: 2.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8366986, upper bound: 1.8065103
time: 2.16 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1447217, 0.8141840, -0.8133291, 1.7165370, -1.8612586, 1.6275132
1: -0.2489406, 0.2825509, -0.7787169, 0.7645046, -1.0134451, 1.0612679
2: -0.3386260, 0.3315427, -0.8600405, 1.0080506, -1.3466766, 1.1915832
3: -0.2383914, 0.1691650, -0.6456788, 0.7764509, -1.0148423, 0.8148438
4: -0.2382457, 0.3219604, -0.8054973, 1.0447588, -1.2830045, 1.1274577
5: -0.3879747, 0.4105630, -1.0054295, 1.1123652, -1.5003400, 1.4159925
6: -0.0047474, 1.2673301, -1.0753813, 1.6602148, -1.6649622, 2.3427114
7: -0.2660576, 0.3833951, -0.9502901, 0.9308218, -1.1968794, 1.3336853
8: -0.2966676, 0.3162770, -0.8730054, 1.0768266, -1.3734941, 1.1892824
9: -0.1616657, 0.2104241, -0.7452032, 0.8580167, -1.0196824, 0.9556273

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8411868, upper bound: 1.8103338
time: 1.99 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8420023, upper bound: 1.8104184
time: 2.14 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1512356, 0.8239231, -1.0014935, 1.9273556, -2.0785913, 1.8254166
1: -0.2547621, 0.2879148, -0.9099270, 0.8902965, -1.1450585, 1.1978419
2: -0.3440895, 0.3379008, -0.9875960, 1.1915798, -1.5356693, 1.3254968
3: -0.2416854, 0.1773226, -0.7595432, 0.9303161, -1.1720015, 0.9368659
4: -0.2438067, 0.3269699, -0.9404840, 1.2707995, -1.5146062, 1.2674539
5: -0.3927096, 0.4175615, -1.1901283, 1.2981069, -1.6908165, 1.6076899
6: -0.0167649, 1.2693782, -1.3315976, 1.7482219, -1.7649868, 2.6009758
7: -0.2732910, 0.3889458, -1.1291368, 1.0696571, -1.3429481, 1.5180826
8: -0.3018544, 0.3257807, -1.0190234, 1.2830817, -1.5849360, 1.3448042
9: -0.1682600, 0.2143402, -0.8930107, 1.0399649, -1.2082249, 1.1073509

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8414214, upper bound: 1.8103104
time: 2.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8422578, upper bound: 1.8103867
time: 2.46 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.9833275, 1.8930256, -2.1312947, 1.9341221
1: -0.3297816, 0.3564569, -0.8961952, 0.8818668, -1.2116485, 1.2526522
2: -0.4150444, 0.4270655, -0.9733252, 1.1776061, -1.5926504, 1.4003906
3: -0.2916944, 0.2636827, -0.7523167, 0.9175037, -1.2091981, 1.0159994
4: -0.3300874, 0.3899961, -0.9256110, 1.2548698, -1.5849571, 1.3156071
5: -0.4548751, 0.5089113, -1.1687191, 1.2888355, -1.7437105, 1.6776303
6: -0.1680785, 1.3109281, -1.2971220, 1.7235489, -1.8916274, 2.6080501
7: -0.3655823, 0.4693472, -1.1144310, 1.0552229, -1.4208052, 1.5837781
8: -0.3765907, 0.4246833, -1.0099509, 1.2683618, -1.6449525, 1.4346342
9: -0.2510976, 0.2932430, -0.8812496, 1.0278685, -1.2789661, 1.1744926

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8576008, upper bound: 1.8223780
time: 2.06 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8576008, upper bound: 1.8290884
time: 2.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.9833275, 1.8930256, -2.0981500, 1.8748279
1: -0.2987980, 0.3325009, -0.8961952, 0.8818668, -1.1806648, 1.2286961
2: -0.3846641, 0.3945331, -0.9733252, 1.1776061, -1.5622702, 1.3678583
3: -0.2706087, 0.2334781, -0.7523167, 0.9175037, -1.1881125, 0.9857948
4: -0.2944325, 0.3679506, -0.9256110, 1.2548698, -1.5493023, 1.2935616
5: -0.4267558, 0.4781152, -1.1687191, 1.2888355, -1.7155913, 1.6468343
6: -0.1026776, 1.2858521, -1.2971220, 1.7235489, -1.8262265, 2.5829740
7: -0.3313026, 0.4365475, -1.1144310, 1.0552229, -1.3865254, 1.5509785
8: -0.3483621, 0.3910702, -1.0099509, 1.2683618, -1.6167239, 1.4010211
9: -0.2218433, 0.2653263, -0.8812496, 1.0278685, -1.2497118, 1.1465759

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8568046, upper bound: 1.8275270
time: 2.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8568046, upper bound: 1.8300619
time: 2.30 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1617692, 0.8587554, -0.6568071, 1.5418290, -1.7035983, 1.5155625
1: -0.2688225, 0.2940106, -0.6697340, 0.6599773, -0.9287997, 0.9637446
2: -0.3592864, 0.3467408, -0.7539785, 0.8554834, -1.2147698, 1.1007193
3: -0.2512638, 0.1853703, -0.5509155, 0.6486378, -0.8999016, 0.7362859
4: -0.2580197, 0.3329864, -0.6935390, 0.8576491, -1.1156688, 1.0265254
5: -0.4089479, 0.4260335, -0.8522582, 0.9565086, -1.3654565, 1.2782917
6: -0.0503951, 1.2839557, -0.8622742, 1.5875899, -1.6379850, 2.1462297
7: -0.2831461, 0.4025550, -0.8017370, 0.8155580, -1.0987041, 1.2042921
8: -0.3127802, 0.3362087, -0.7530220, 0.9051627, -1.2179428, 1.0892307
9: -0.1734019, 0.2178102, -0.6232421, 0.7068179, -0.8802198, 0.8410524

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8306336, upper bound: 1.8055483
time: 1.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8321000, upper bound: 1.8059429
time: 2.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1683012, 0.8686923, -0.8376607, 1.7439908, -1.9122920, 1.7063529
1: -0.2746186, 0.2994523, -0.7955112, 0.7805547, -1.0551733, 1.0949636
2: -0.3647394, 0.3532646, -0.8763928, 1.0316062, -1.3963456, 1.2296574
3: -0.2546288, 0.1934465, -0.6601479, 0.7964090, -1.0510378, 0.8535944
4: -0.2637104, 0.3379016, -0.8229948, 1.0743488, -1.3380592, 1.1608964
5: -0.4137072, 0.4329888, -1.0295265, 1.1347960, -1.5485032, 1.4625152
6: -0.0625393, 1.2873634, -1.1086504, 1.6712414, -1.7337807, 2.3960137
7: -0.2903322, 0.4082829, -0.9735131, 0.9485780, -1.2389102, 1.3817960
8: -0.3182665, 0.3455703, -0.8915544, 1.1030877, -1.4213541, 1.2371247
9: -0.1801065, 0.2216410, -0.7649800, 0.8814525, -1.0615590, 0.9866210

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8311623, upper bound: 1.8055465
time: 2.45 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8326200, upper bound: 1.8059430
time: 1.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1465747, 0.8076330, -0.7240772, 1.6119542, -1.7585289, 1.5317101
1: -0.2486294, 0.2856561, -0.7160007, 0.7061043, -0.9547337, 1.0016568
2: -0.3373639, 0.3346429, -0.7985218, 0.9224098, -1.2597737, 1.1331648
3: -0.2375437, 0.1735912, -0.5917774, 0.7044508, -0.9419944, 0.7653687
4: -0.2374558, 0.3252537, -0.7408864, 0.9404395, -1.1778953, 1.0661402
5: -0.3853159, 0.4162081, -0.9168366, 1.0259539, -1.4112698, 1.3330446
6: -0.0020253, 1.2718351, -0.9501106, 1.6125245, -1.6145499, 2.2219458
7: -0.2691860, 0.3826792, -0.8666574, 0.8642146, -1.1334006, 1.2493365
8: -0.2998148, 0.3220846, -0.8033180, 0.9805225, -1.2803373, 1.1254026
9: -0.1654713, 0.2142726, -0.6770204, 0.7738357, -0.9393070, 0.8912930

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8383001, upper bound: 1.8103173
time: 2.49 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8393475, upper bound: 1.8104090
time: 2.03 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1527571, 0.8170431, -0.9050493, 1.8140879, -1.9668450, 1.7220924
1: -0.2542071, 0.2907304, -0.8421152, 0.8268082, -1.0810153, 1.1328455
2: -0.3425732, 0.3406829, -0.9212248, 1.0986670, -1.4412403, 1.2619078
3: -0.2407127, 0.1813438, -0.7013432, 0.8523324, -1.0930451, 0.8826870
4: -0.2427678, 0.3299884, -0.8706009, 1.1573404, -1.4001082, 1.2005894
5: -0.3898918, 0.4228162, -1.0944794, 1.2044262, -1.5943180, 1.5172956
6: -0.0135211, 1.2734371, -1.1968210, 1.6962959, -1.7098169, 2.4702582
7: -0.2760799, 0.3879716, -1.0385648, 0.9977654, -1.2738452, 1.4265364
8: -0.3047269, 0.3310803, -0.9419907, 1.1786039, -1.4833308, 1.2730711
9: -0.1717396, 0.2179645, -0.8188590, 0.9486030, -1.1203426, 1.0368235

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8386500, upper bound: 1.8102919
time: 2.23 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8397608, upper bound: 1.8103775
time: 2.30 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -0.8746941, 1.7660595, -2.0226147, 1.8491343
1: -0.3448424, 0.3715456, -0.8197067, 0.8105962, -1.1554387, 1.1912524
2: -0.4289511, 0.4466715, -0.8984438, 1.0732567, -1.5022079, 1.3451153
3: -0.3015637, 0.2813208, -0.6843544, 0.8299475, -1.1315112, 0.9656752
4: -0.3472072, 0.4048828, -0.8468271, 1.1276919, -1.4748991, 1.2517099
5: -0.4669372, 0.5324135, -1.0609493, 1.1835604, -1.6504976, 1.5933628
6: -0.1975362, 1.3229021, -1.1451206, 1.6648192, -1.8623555, 2.4680228
7: -0.3851250, 0.4853222, -1.0126957, 0.9740661, -1.3591912, 1.4980178
8: -0.3923242, 0.4465339, -0.9225860, 1.1508989, -1.5432231, 1.3691199
9: -0.2674738, 0.3125915, -0.7982261, 0.9254314, -1.1929052, 1.1108176

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8522478, upper bound: 1.8204159
time: 1.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8522478, upper bound: 1.8287593
time: 2.04 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2067769, 0.8844675, -0.8746941, 1.7660595, -1.9728364, 1.7591616
1: -0.2982824, 0.3353040, -0.8197067, 0.8105962, -1.1088786, 1.1550107
2: -0.3830329, 0.3976636, -0.8984438, 1.0732567, -1.4562896, 1.2961074
3: -0.2701186, 0.2365419, -0.6843544, 0.8299475, -1.1000661, 0.9208962
4: -0.2941514, 0.3709590, -0.8468271, 1.1276919, -1.4218433, 1.2177862
5: -0.4239522, 0.4835648, -1.0609493, 1.1835604, -1.6075126, 1.5445142
6: -0.0985513, 1.2864097, -1.1451206, 1.6648192, -1.7633705, 2.4315305
7: -0.3340706, 0.4360552, -1.0126957, 0.9740661, -1.3081367, 1.4487510
8: -0.3514939, 0.3950947, -0.9225860, 1.1508989, -1.5023928, 1.3176806
9: -0.2252686, 0.2706831, -0.7982261, 0.9254314, -1.1507000, 1.0689092

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8550277, upper bound: 1.8271880
time: 2.11 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8550277, upper bound: 1.8300520
time: 2.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.9833275, 1.8930256, -0.2382691, 0.9507946, -1.9341221, 2.1312947
1: -0.8961952, 0.8818668, -0.3297816, 0.3564569, -1.2526522, 1.2116485
2: -0.9733252, 1.1776061, -0.4150444, 0.4270655, -1.4003906, 1.5926504
3: -0.7523167, 0.9175037, -0.2916944, 0.2636827, -1.0159994, 1.2091981
4: -0.9256110, 1.2548698, -0.3300874, 0.3899961, -1.3156071, 1.5849571
5: -1.1687191, 1.2888355, -0.4548751, 0.5089113, -1.6776303, 1.7437105
6: -1.2971220, 1.7235489, -0.1680785, 1.3109281, -2.6080501, 1.8916274
7: -1.1144310, 1.0552229, -0.3655823, 0.4693472, -1.5837781, 1.4208052
8: -1.0099509, 1.2683618, -0.3765907, 0.4246833, -1.4346342, 1.6449525
9: -0.8812496, 1.0278685, -0.2510976, 0.2932430, -1.1744926, 1.2789661

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8286488, upper bound: 1.8407838
time: 2.17 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8286488, upper bound: 1.8412186
time: 2.06 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.9833275, 1.8930256, -0.6409820, 1.5177834, -2.5011110, 2.5340075
1: -0.8961952, 0.8818668, -0.6670967, 0.6740783, -1.5702735, 1.5489635
2: -0.9733252, 1.1776061, -0.7308092, 0.8434644, -1.8167896, 1.9084153
3: -0.7523167, 0.9175037, -0.5171071, 0.6514361, -1.4037528, 1.4346108
4: -0.9256110, 1.2548698, -0.7264148, 0.6882318, -1.6138428, 1.9812846
5: -1.1687191, 1.2888355, -0.7533011, 0.9482372, -2.1169562, 2.0421367
6: -1.2971220, 1.7235489, -0.8369410, 1.6103916, -2.9075136, 2.5604899
7: -1.1144310, 1.0552229, -0.7920405, 0.8317302, -1.9461613, 1.8472633
8: -1.0099509, 1.2683618, -0.7391541, 0.8698435, -1.8797944, 2.0075159
9: -0.8812496, 1.0278685, -0.6328616, 0.6930045, -1.5742540, 1.6607301

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8286488, upper bound: 1.8407838
time: 2.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8286488, upper bound: 1.8412186
time: 2.39 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.9833275, 1.8930256, -0.2051244, 0.8915004, -1.8748279, 2.0981500
1: -0.8961952, 0.8818668, -0.2987980, 0.3325009, -1.2286961, 1.1806648
2: -0.9733252, 1.1776061, -0.3846641, 0.3945331, -1.3678583, 1.5622702
3: -0.7523167, 0.9175037, -0.2706087, 0.2334781, -0.9857948, 1.1881125
4: -0.9256110, 1.2548698, -0.2944325, 0.3679506, -1.2935616, 1.5493023
5: -1.1687191, 1.2888355, -0.4267558, 0.4781152, -1.6468343, 1.7155913
6: -1.2971220, 1.7235489, -0.1026776, 1.2858521, -2.5829740, 1.8262265
7: -1.1144310, 1.0552229, -0.3313026, 0.4365475, -1.5509785, 1.3865254
8: -1.0099509, 1.2683618, -0.3483621, 0.3910702, -1.4010211, 1.6167239
9: -0.8812496, 1.0278685, -0.2218433, 0.2653263, -1.1465759, 1.2497118

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8297394, upper bound: 1.8430532
time: 3.11 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8297394, upper bound: 1.8433551
time: 2.02 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.9833275, 1.8930256, -0.6187497, 1.4702761, -2.4536037, 2.5117755
1: -0.8961952, 0.8818668, -0.6452557, 0.6584682, -1.5546634, 1.5271225
2: -0.9733252, 1.1776061, -0.7093706, 0.8218896, -1.7952148, 1.8869767
3: -0.7523167, 0.9175037, -0.5023815, 0.6309984, -1.3833151, 1.4198852
4: -0.9256110, 1.2548698, -0.7011853, 0.6743190, -1.5999300, 1.9560552
5: -1.1687191, 1.2888355, -0.7321635, 0.9295232, -2.0982423, 2.0209990
6: -1.2971220, 1.7235489, -0.7871232, 1.5770494, -2.8741713, 2.5106721
7: -1.1144310, 1.0552229, -0.7688296, 0.8095052, -1.9239362, 1.8240526
8: -1.0099509, 1.2683618, -0.7128940, 0.8481259, -1.8580768, 1.9812558
9: -0.8812496, 1.0278685, -0.6127145, 0.6747947, -1.5560443, 1.6405830

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8297394, upper bound: 1.8430532
time: 2.16 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8297394, upper bound: 1.8433551
time: 1.77 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.19 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8937388, upper bound: 1.8863091
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8931445, upper bound: 1.8863106
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8937388, upper bound: 1.8863091
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8931445, upper bound: 1.8863106
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8937436, upper bound: 1.8871205
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8931764, upper bound: 1.8871236
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8937436, upper bound: 1.8871205
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8931764, upper bound: 1.8871235
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8956502, upper bound: 1.8872369
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8956502, upper bound: 1.8878554
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8956502, upper bound: 1.8872369
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8956502, upper bound: 1.8878554
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8945125, upper bound: 1.8878040
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8945125, upper bound: 1.8883937
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8945125, upper bound: 1.8878040
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8945125, upper bound: 1.8883937
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8732932, upper bound: 1.8754558
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8759964, upper bound: 1.8757412
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8732937, upper bound: 1.8745655
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8759996, upper bound: 1.8747944
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8868715, upper bound: 1.8865594
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8865121, upper bound: 1.8865610
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8868715, upper bound: 1.8871052
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8865121, upper bound: 1.8871076
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8868402, upper bound: 1.8870008
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8862572, upper bound: 1.8870034
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8873423, upper bound: 1.8870456
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8873423, upper bound: 1.8870456
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8874072, upper bound: 1.8877897
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8871057, upper bound: 1.8877924
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8880674, upper bound: 1.8877609
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8880674, upper bound: 1.8883800
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8322606, upper bound: 1.8046867
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8360871, upper bound: 1.8065103
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8328922, upper bound: 1.8046606
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8366986, upper bound: 1.8065103
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8411868, upper bound: 1.8103338
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8420023, upper bound: 1.8104184
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8414214, upper bound: 1.8103104
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8422578, upper bound: 1.8103867
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8576008, upper bound: 1.8223780
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8576008, upper bound: 1.8290884
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8568046, upper bound: 1.8275270
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8568046, upper bound: 1.8300619
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8306336, upper bound: 1.8055483
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8321000, upper bound: 1.8059429
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8311623, upper bound: 1.8055465
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8326200, upper bound: 1.8059430
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8383001, upper bound: 1.8103173
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8393475, upper bound: 1.8104090
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8386500, upper bound: 1.8102919
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8397608, upper bound: 1.8103775
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8522478, upper bound: 1.8204159
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8522478, upper bound: 1.8287593
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8550277, upper bound: 1.8271880
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8550277, upper bound: 1.8300520
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8286488, upper bound: 1.8407838
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8286488, upper bound: 1.8412186
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8286488, upper bound: 1.8407838
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8286488, upper bound: 1.8412186
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8297394, upper bound: 1.8430532
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8297394, upper bound: 1.8433551
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8297394, upper bound: 1.8430532
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 6, lower bound: -1.8297394, upper bound: 1.8433551

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1082251, 0.6951822, -0.1779314, 0.8770279, -0.9852529, 0.8731136
1: -0.1978365, 0.2282948, -0.2813197, 0.3078374, -0.5056739, 0.5096145
2: -0.2850994, 0.2537735, -0.3700670, 0.3639101, -0.6490095, 0.6238405
3: -0.1996000, 0.1119908, -0.2591533, 0.2047832, -0.4043832, 0.3711441
4: -0.1797125, 0.2738874, -0.2729360, 0.3439451, -0.5236576, 0.5468234
5: -0.3438171, 0.3433841, -0.4174722, 0.4396802, -0.7834972, 0.7608564
6: 0.1268533, 1.2440593, -0.0747913, 1.2826009, -1.1557477, 1.3188505
7: -0.1907063, 0.3347956, -0.3012540, 0.4167665, -0.6074728, 0.6360496
8: -0.2371492, 0.2376753, -0.3251692, 0.3565403, -0.5936895, 0.5628445
9: -0.1059172, 0.1687125, -0.1931310, 0.2309379, -0.3368551, 0.3618435

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8858838, upper bound: 1.8840776
time: 2.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8859925, upper bound: 1.8851126
time: 2.41 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1407404, 0.8309262, -0.1866159, 0.8876849, -1.0284253, 1.0175420
1: -0.2509136, 0.2736502, -0.2883473, 0.3147541, -0.5656677, 0.5619975
2: -0.3432438, 0.3232007, -0.3766321, 0.3728935, -0.7161373, 0.6998328
3: -0.2408333, 0.1566080, -0.2638715, 0.2132109, -0.4540443, 0.4204795
4: -0.2400817, 0.3148011, -0.2811660, 0.3505310, -0.5906128, 0.5959672
5: -0.3959285, 0.3980896, -0.4227178, 0.4495309, -0.8454593, 0.8208075
6: -0.0144866, 1.2744644, -0.0884711, 1.2865138, -1.3010004, 1.3629354
7: -0.2569302, 0.3856127, -0.3104996, 0.4243320, -0.6812621, 0.6961123
8: -0.2963239, 0.3058109, -0.3326605, 0.3663158, -0.6626397, 0.6384714
9: -0.1516153, 0.2024392, -0.2013812, 0.2398216, -0.3914369, 0.4038205

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8852058, upper bound: 1.8840763
time: 2.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8853412, upper bound: 1.8851109
time: 2.86 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1082251, 0.6951822, -0.1975220, 0.9007849, -1.0090100, 0.8927042
1: -0.1978365, 0.2282948, -0.2973751, 0.3237416, -0.5215782, 0.5256699
2: -0.2850994, 0.2537735, -0.3849719, 0.3847117, -0.6698111, 0.6387454
3: -0.1996000, 0.1119908, -0.2697721, 0.2236702, -0.4232701, 0.3817629
4: -0.1797125, 0.2738874, -0.2910907, 0.3596238, -0.5393363, 0.5649781
5: -0.3438171, 0.3433841, -0.4295750, 0.4646887, -0.8085058, 0.7729591
6: 0.1268533, 1.2440593, -0.1063361, 1.2945962, -1.1677430, 1.3503954
7: -0.1907063, 0.3347956, -0.3221949, 0.4336010, -0.6243072, 0.6569905
8: -0.2371492, 0.2376753, -0.3420867, 0.3795874, -0.6167366, 0.5797620
9: -0.1059172, 0.1687125, -0.2106369, 0.2514287, -0.3573459, 0.3793494

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8826873, upper bound: 1.8727699
time: 2.30 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8831230, upper bound: 1.8748591
time: 2.84 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1407404, 0.8309262, -0.2057341, 0.9109533, -1.0516938, 1.0366603
1: -0.2509136, 0.2736502, -0.3040311, 0.3303136, -0.5812272, 0.5776813
2: -0.3432438, 0.3232007, -0.3911605, 0.3932514, -0.7364951, 0.7143612
3: -0.2408333, 0.1566080, -0.2742194, 0.2316464, -0.4724797, 0.4308274
4: -0.2400817, 0.3148011, -0.2989313, 0.3658243, -0.6059060, 0.6137323
5: -0.3959285, 0.3980896, -0.4347294, 0.4739327, -0.8698611, 0.8328190
6: -0.0144866, 1.2744644, -0.1191752, 1.2983656, -1.3128521, 1.3936396
7: -0.2569302, 0.3856127, -0.3309119, 0.4408250, -0.6977552, 0.7165245
8: -0.2963239, 0.3058109, -0.3491522, 0.3887876, -0.6851115, 0.6549631
9: -0.1516153, 0.2024392, -0.2184976, 0.2598035, -0.4114189, 0.4209368

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8819805, upper bound: 1.8727677
time: 2.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8825025, upper bound: 1.8748561
time: 2.23 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1127624, 0.6586977, -0.2150804, 0.9212438, -1.0340062, 0.8737780
1: -0.1841899, 0.2248551, -0.3109617, 0.3378776, -0.5220675, 0.5358167
2: -0.2701199, 0.2488945, -0.3975527, 0.4028647, -0.6729846, 0.6464472
3: -0.1905187, 0.1072814, -0.2791124, 0.2411552, -0.4316739, 0.3863938
4: -0.1659738, 0.2706232, -0.3078786, 0.3725088, -0.5384827, 0.5785018
5: -0.3265649, 0.3405934, -0.4400503, 0.4828545, -0.8094194, 0.7806437
6: 0.1598627, 1.2474368, -0.1316835, 1.2991772, -1.1393144, 1.3791203
7: -0.1840844, 0.3212462, -0.3409778, 0.4488276, -0.6329120, 0.6622240
8: -0.2319536, 0.2322178, -0.3565967, 0.3987262, -0.6306798, 0.5888145
9: -0.1087218, 0.1686768, -0.2288533, 0.2696380, -0.3783599, 0.3975301

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8962803, upper bound: 1.8964987
time: 2.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8962803, upper bound: 1.8964987
time: 2.37 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1320025, 0.7922729, -0.2237346, 0.9321672, -1.0641698, 1.0160075
1: -0.2365903, 0.2704820, -0.3179772, 0.3448228, -0.5814131, 0.5884592
2: -0.3274387, 0.3172338, -0.4040706, 0.4119112, -0.7393498, 0.7213044
3: -0.2310056, 0.1521887, -0.2838015, 0.2495669, -0.4805726, 0.4359902
4: -0.2256765, 0.3115505, -0.3161477, 0.3790524, -0.6047289, 0.6276982
5: -0.3784527, 0.3951966, -0.4455283, 0.4926402, -0.8710929, 0.8407248
6: 0.0216799, 1.2637713, -0.1452520, 1.3034813, -1.2818015, 1.4090233
7: -0.2496336, 0.3717154, -0.3501682, 0.4564676, -0.7061012, 0.7218837
8: -0.2853023, 0.2997836, -0.3640393, 0.4084372, -0.6937395, 0.6638230
9: -0.1494163, 0.2017526, -0.2371470, 0.2784684, -0.4278847, 0.4388996

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8956616, upper bound: 1.8965051
time: 2.14 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8956616, upper bound: 1.8965051
time: 2.16 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1127624, 0.6586977, -0.2342491, 0.9458109, -1.0585734, 0.8929467
1: -0.1841899, 0.2248551, -0.3267307, 0.3535410, -0.5377309, 0.5515858
2: -0.2701199, 0.2488945, -0.4121493, 0.4234117, -0.6935316, 0.6610438
3: -0.1905187, 0.1072814, -0.2894704, 0.2596480, -0.4501668, 0.3967518
4: -0.1659738, 0.2706232, -0.3257644, 0.3878434, -0.5538173, 0.5963876
5: -0.3265649, 0.3405934, -0.4523515, 0.5073977, -0.8339626, 0.7929449
6: 0.1598627, 1.2474368, -0.1626372, 1.3115113, -1.1516485, 1.4100740
7: -0.1840844, 0.3212462, -0.3614312, 0.4655858, -0.6496702, 0.6826774
8: -0.2319536, 0.2322178, -0.3731312, 0.4212552, -0.6532088, 0.6053489
9: -0.1087218, 0.1686768, -0.2460295, 0.2896376, -0.3983595, 0.4147063

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8922086, upper bound: 1.8865752
time: 2.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8922086, upper bound: 1.8871205
time: 2.45 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1320025, 0.7922729, -0.2424576, 0.9563271, -1.0883296, 1.0347306
1: -0.2365903, 0.2704820, -0.3333829, 0.3601241, -0.5967144, 0.6038649
2: -0.3274387, 0.3172338, -0.4183201, 0.4319802, -0.7594188, 0.7355540
3: -0.2310056, 0.1521887, -0.2939077, 0.2676300, -0.4986357, 0.4460965
4: -0.2256765, 0.3115505, -0.3336310, 0.3940363, -0.6197128, 0.6451815
5: -0.3784527, 0.3951966, -0.4576597, 0.5166200, -0.8950726, 0.8528563
6: 0.0216799, 1.2637713, -0.1754807, 1.3156424, -1.2939625, 1.4392519
7: -0.2496336, 0.3717154, -0.3701386, 0.4728486, -0.7224821, 0.7418541
8: -0.2853023, 0.2997836, -0.3801849, 0.4304430, -0.7157453, 0.6799686
9: -0.1494163, 0.2017526, -0.2539164, 0.2980030, -0.4474193, 0.4556690

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8917609, upper bound: 1.8865779
time: 2.26 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8917609, upper bound: 1.8871236
time: 2.00 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2243731, 0.9671444, -0.1672815, 0.8462785, -1.0706517, 1.1344260
1: -0.3271942, 0.3412869, -0.2684518, 0.3018621, -0.6290563, 0.6097386
2: -0.4163548, 0.4090855, -0.3565905, 0.3549665, -0.7713213, 0.7656760
3: -0.2904871, 0.2458294, -0.2501785, 0.1964472, -0.4869343, 0.4960079
4: -0.3241761, 0.3759019, -0.2587461, 0.3386775, -0.6628537, 0.6346481
5: -0.4602108, 0.4860530, -0.4037264, 0.4342439, -0.8944547, 0.8897794
6: -0.1745159, 1.3373069, -0.0446550, 1.2734960, -1.4480119, 1.3819618
7: -0.3487638, 0.4645955, -0.2908118, 0.4036873, -0.7524511, 0.7554073
8: -0.3782752, 0.4053175, -0.3151361, 0.3478192, -0.7260944, 0.7204536
9: -0.2305420, 0.2695221, -0.1854764, 0.2257000, -0.4562420, 0.4549984

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8964987, upper bound: 1.8962803
time: 2.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8965051, upper bound: 1.8956616
time: 2.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2134842, 0.9210240, -0.2051244, 0.8915004, -1.1049845, 1.1261485
1: -0.3100782, 0.3364041, -0.2987980, 0.3325009, -0.6425791, 0.6352021
2: -0.3969255, 0.4010248, -0.3846641, 0.3945331, -0.7914587, 0.7856889
3: -0.2785302, 0.2393028, -0.2706087, 0.2334781, -0.5120083, 0.5099115
4: -0.3066370, 0.3711747, -0.2944325, 0.3679506, -0.6745875, 0.6656072
5: -0.4397345, 0.4809099, -0.4267558, 0.4781152, -0.9178497, 0.9076657
6: -0.1306572, 1.3001194, -0.1026776, 1.2858521, -1.4165093, 1.4027970
7: -0.3390956, 0.4477824, -0.3313026, 0.4365475, -0.7756431, 0.7790850
8: -0.3558619, 0.3968179, -0.3483621, 0.3910702, -0.7469321, 0.7451800
9: -0.2267661, 0.2674844, -0.2218433, 0.2653263, -0.4920924, 0.4893277

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8967487, upper bound: 1.8969979
time: 3.14 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8967487, upper bound: 1.8969979
time: 2.46 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2243731, 0.9671444, -0.1694376, 0.8388355, -1.0632086, 1.1365820
1: -0.3271942, 0.3412869, -0.2681902, 0.3050637, -0.6322579, 0.6094771
2: -0.4163548, 0.4090855, -0.3551775, 0.3586025, -0.7749574, 0.7642630
3: -0.2904871, 0.2458294, -0.2497055, 0.1999865, -0.4904736, 0.4955349
4: -0.3241761, 0.3759019, -0.2588218, 0.3420937, -0.6662698, 0.6347238
5: -0.4602108, 0.4860530, -0.4008318, 0.4402800, -0.9004908, 0.8868848
6: -0.1745159, 1.3373069, -0.0411611, 1.2761184, -1.4506342, 1.3784679
7: -0.3487638, 0.4645955, -0.2940759, 0.4034359, -0.7521997, 0.7586713
8: -0.3782752, 0.4053175, -0.3186762, 0.3524678, -0.7307431, 0.7239937
9: -0.2305420, 0.2695221, -0.1893898, 0.2316148, -0.4621568, 0.4589119

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8946934, upper bound: 1.8870974
time: 2.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8947441, upper bound: 1.8867529
time: 2.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2134842, 0.9210240, -0.2067769, 0.8844675, -1.0979517, 1.1278009
1: -0.3100782, 0.3364041, -0.2982824, 0.3353040, -0.6453822, 0.6346865
2: -0.3969255, 0.4010248, -0.3830329, 0.3976636, -0.7945892, 0.7840577
3: -0.2785302, 0.2393028, -0.2701186, 0.2365419, -0.5150720, 0.5094213
4: -0.3066370, 0.3711747, -0.2941514, 0.3709590, -0.6775960, 0.6653260
5: -0.4397345, 0.4809099, -0.4239522, 0.4835648, -0.9232993, 0.9048622
6: -0.1306572, 1.3001194, -0.0985513, 1.2864097, -1.4170669, 1.3986707
7: -0.3390956, 0.4477824, -0.3340706, 0.4360552, -0.7751509, 0.7818531
8: -0.3558619, 0.3968179, -0.3514939, 0.3950947, -0.7509565, 0.7483118
9: -0.2267661, 0.2674844, -0.2252686, 0.2706831, -0.4974492, 0.4927530

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8943947, upper bound: 1.8877212
time: 2.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8943947, upper bound: 1.8878554
time: 2.49 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1881544, 0.9007847, -0.1672815, 0.8462785, -1.0344329, 1.0680662
1: -0.2925890, 0.3149697, -0.2684518, 0.3018621, -0.5944511, 0.5834215
2: -0.3818289, 0.3739185, -0.3565905, 0.3549665, -0.7367954, 0.7305090
3: -0.2668294, 0.2129367, -0.2501785, 0.1964472, -0.4632766, 0.4631152
4: -0.2844892, 0.3514729, -0.2587461, 0.3386775, -0.6231668, 0.6102190
5: -0.4286769, 0.4521877, -0.4037264, 0.4342439, -0.8629208, 0.8559141
6: -0.1011343, 1.3000320, -0.0446550, 1.2734960, -1.3746303, 1.3446870
7: -0.3113961, 0.4279875, -0.2908118, 0.4036873, -0.7150835, 0.7187994
8: -0.3383414, 0.3679772, -0.3151361, 0.3478192, -0.6861606, 0.6831133
9: -0.1988900, 0.2385730, -0.1854764, 0.2257000, -0.4245900, 0.4240493

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8961928, upper bound: 1.8966231
time: 2.27 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8961952, upper bound: 1.8959426
time: 2.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1789902, 0.8608237, -0.2051244, 0.8915004, -1.0704906, 1.0659481
1: -0.2779647, 0.3114245, -0.2987980, 0.3325009, -0.6104655, 0.6102225
2: -0.3655255, 0.3672579, -0.3846641, 0.3945331, -0.7600586, 0.7519220
3: -0.2566280, 0.2077801, -0.2706087, 0.2334781, -0.4901061, 0.4783888
4: -0.2697608, 0.3480240, -0.2944325, 0.3679506, -0.6377113, 0.6424565
5: -0.4110833, 0.4486107, -0.4267558, 0.4781152, -0.8891985, 0.8753664
6: -0.0632123, 1.2787700, -0.1026776, 1.2858521, -1.3490644, 1.3814476
7: -0.3033165, 0.4138650, -0.3313026, 0.4365475, -0.7398640, 0.7451676
8: -0.3260965, 0.3615586, -0.3483621, 0.3910702, -0.7171667, 0.7099207
9: -0.1962599, 0.2380444, -0.2218433, 0.2653263, -0.4615861, 0.4598877

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8964859, upper bound: 1.8972860
time: 2.17 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8964859, upper bound: 1.8972860
time: 2.26 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1881544, 0.9007847, -0.1694376, 0.8388355, -1.0269898, 1.0702224
1: -0.2925890, 0.3149697, -0.2681902, 0.3050637, -0.5976527, 0.5831599
2: -0.3818289, 0.3739185, -0.3551775, 0.3586025, -0.7404314, 0.7290960
3: -0.2668294, 0.2129367, -0.2497055, 0.1999865, -0.4668159, 0.4626422
4: -0.2844892, 0.3514729, -0.2588218, 0.3420937, -0.6265829, 0.6102948
5: -0.4286769, 0.4521877, -0.4008318, 0.4402800, -0.8689569, 0.8530195
6: -0.1011343, 1.3000320, -0.0411611, 1.2761184, -1.3772527, 1.3411931
7: -0.3113961, 0.4279875, -0.2940759, 0.4034359, -0.7148321, 0.7220634
8: -0.3383414, 0.3679772, -0.3186762, 0.3524678, -0.6908092, 0.6866534
9: -0.1988900, 0.2385730, -0.1893898, 0.2316148, -0.4305048, 0.4279628

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8936766, upper bound: 1.8877123
time: 2.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8936795, upper bound: 1.8873748
time: 2.42 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1789902, 0.8608237, -0.2067769, 0.8844675, -1.0634577, 1.0676006
1: -0.2779647, 0.3114245, -0.2982824, 0.3353040, -0.6132686, 0.6097069
2: -0.3655255, 0.3672579, -0.3830329, 0.3976636, -0.7631892, 0.7502908
3: -0.2566280, 0.2077801, -0.2701186, 0.2365419, -0.4931699, 0.4778987
4: -0.2697608, 0.3480240, -0.2941514, 0.3709590, -0.6407198, 0.6421754
5: -0.4110833, 0.4486107, -0.4239522, 0.4835648, -0.8946481, 0.8725629
6: -0.0632123, 1.2787700, -0.0985513, 1.2864097, -1.3496220, 1.3773212
7: -0.3033165, 0.4138650, -0.3340706, 0.4360552, -0.7393717, 0.7479357
8: -0.3260965, 0.3615586, -0.3514939, 0.3950947, -0.7211912, 0.7130525
9: -0.1962599, 0.2380444, -0.2252686, 0.2706831, -0.4669429, 0.4633130

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8933251, upper bound: 1.8882962
time: 2.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8933251, upper bound: 1.8883937
time: 2.54 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1112843, 0.6362125, -0.1209539, 0.7751015, -0.8863859, 0.7571664
1: -0.1686162, 0.2142441, -0.2280134, 0.2458148, -0.4144310, 0.4422575
2: -0.2563438, 0.2271663, -0.3175261, 0.2918159, -0.5481598, 0.5446924
3: -0.1772667, 0.1021034, -0.2236970, 0.1291711, -0.3064378, 0.3258004
4: -0.1520254, 0.2592115, -0.2135950, 0.2934069, -0.4454323, 0.4728065
5: -0.3189214, 0.3214731, -0.3735233, 0.3726717, -0.6915932, 0.6949965
6: 0.1931952, 1.2438983, 0.0466594, 1.2582815, -1.0650862, 1.1972389
7: -0.1684165, 0.3052878, -0.2230822, 0.3643223, -0.5327388, 0.5283701
8: -0.2174562, 0.2125836, -0.2685527, 0.2740374, -0.4914935, 0.4811362
9: -0.1071599, 0.1597236, -0.1266651, 0.1863118, -0.2934716, 0.2863886

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8629564, upper bound: 1.8603913
time: 2.18 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8641842, upper bound: 1.8660429
time: 2.50 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1132220, 0.6992426, -0.1251974, 0.7892564, -0.9024783, 0.8244400
1: -0.1998336, 0.2324174, -0.2334965, 0.2503924, -0.4502260, 0.4659139
2: -0.2872006, 0.2626909, -0.3238252, 0.2981612, -0.5853618, 0.5865161
3: -0.2021513, 0.1153253, -0.2279292, 0.1330135, -0.3351648, 0.3432545
4: -0.1825870, 0.2789294, -0.2197208, 0.2974117, -0.4799986, 0.4986501
5: -0.3442895, 0.3532712, -0.3791728, 0.3777044, -0.7219939, 0.7324440
6: 0.1204023, 1.2524755, 0.0317238, 1.2628402, -1.1424379, 1.2207518
7: -0.1971813, 0.3363692, -0.2292829, 0.3697503, -0.5669315, 0.5656521
8: -0.2448000, 0.2464031, -0.2754644, 0.2804538, -0.5252538, 0.5218676
9: -0.1097746, 0.1748872, -0.1309957, 0.1895653, -0.2993399, 0.3058828

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8657129, upper bound: 1.8605790
time: 2.11 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8669637, upper bound: 1.8663260
time: 2.25 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1117204, 0.6408149, -0.1827212, 0.9071758, -1.0188961, 0.8235361
1: -0.1721065, 0.2163907, -0.2916505, 0.3092095, -0.4813159, 0.5080412
2: -0.2594104, 0.2309344, -0.3820024, 0.3670265, -0.6264368, 0.6129368
3: -0.1801252, 0.1030525, -0.2662030, 0.2060972, -0.3862224, 0.3692555
4: -0.1548082, 0.2614862, -0.2825064, 0.3459738, -0.5007820, 0.5439926
5: -0.3208741, 0.3255256, -0.4311513, 0.4431487, -0.7640228, 0.7566768
6: 0.1862888, 1.2451363, -0.1024727, 1.3082378, -1.1219490, 1.3476090
7: -0.1712866, 0.3087782, -0.3047265, 0.4263922, -0.5976788, 0.6135048
8: -0.2204043, 0.2160773, -0.3388411, 0.3602649, -0.5806692, 0.5549185
9: -0.1075944, 0.1613129, -0.1911864, 0.2299892, -0.3375836, 0.3524994

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8629657, upper bound: 1.8596354
time: 2.38 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8641842, upper bound: 1.8653880
time: 2.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1136560, 0.7083343, -0.1903817, 0.9212471, -1.0349032, 0.8987160
1: -0.2033044, 0.2347135, -0.2989387, 0.3148934, -0.5181977, 0.5336521
2: -0.2909364, 0.2677953, -0.3893542, 0.3748108, -0.6657472, 0.6571494
3: -0.2050171, 0.1174641, -0.2713235, 0.2132050, -0.4182220, 0.3887875
4: -0.1866145, 0.2815190, -0.2905430, 0.3515429, -0.5381573, 0.5720620
5: -0.3475243, 0.3572122, -0.4374205, 0.4516158, -0.7991401, 0.7946327
6: 0.1111180, 1.2538978, -0.1188827, 1.3161615, -1.2050436, 1.3727804
7: -0.2013312, 0.3398142, -0.3130586, 0.4338535, -0.6351847, 0.6528728
8: -0.2485430, 0.2512559, -0.3472999, 0.3687006, -0.6172435, 0.5985558
9: -0.1128204, 0.1772946, -0.1976074, 0.2366934, -0.3495138, 0.3749020

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8657190, upper bound: 1.8598276
time: 2.03 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8669649, upper bound: 1.8656382
time: 3.29 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1166757, 0.6639405, -0.2910718, 1.0898170, -1.2064927, 0.9550123
1: -0.1867247, 0.2285658, -0.3911481, 0.3909040, -0.5776287, 0.6197139
2: -0.2726291, 0.2572156, -0.4797208, 0.4760216, -0.7486507, 0.7369363
3: -0.1932868, 0.1103589, -0.3340391, 0.3064844, -0.4997712, 0.4443980
4: -0.1692145, 0.2753257, -0.3959187, 0.4234065, -0.5926210, 0.6712444
5: -0.3275969, 0.3487661, -0.5186158, 0.5567189, -0.8843158, 0.8673820
6: 0.1520091, 1.2557375, -0.3124511, 1.4189019, -1.2668928, 1.5681887
7: -0.1903213, 0.3234246, -0.4180807, 0.5313072, -0.7216285, 0.7415053
8: -0.2385489, 0.2405939, -0.4519744, 0.4771902, -0.7157391, 0.6925684
9: -0.1116305, 0.1743712, -0.2862671, 0.3285313, -0.4401618, 0.4606383

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8771831, upper bound: 1.8721502
time: 2.30 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8784567, upper bound: 1.8781099
time: 2.37 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1326349, 0.7848369, -0.3002143, 1.1016999, -1.2343348, 1.0850512
1: -0.2355388, 0.2731448, -0.3985116, 0.3982355, -0.6337743, 0.6716564
2: -0.3254093, 0.3197970, -0.4865489, 0.4855840, -0.8109933, 0.8063459
3: -0.2298462, 0.1554930, -0.3389633, 0.3153691, -0.5452153, 0.4944563
4: -0.2244644, 0.3141501, -0.4046193, 0.4303246, -0.6547890, 0.7187694
5: -0.3750610, 0.4001259, -0.5246367, 0.5670459, -0.9421070, 0.9247626
6: 0.0254992, 1.2685649, -0.3268917, 1.4245892, -1.3990901, 1.5954566
7: -0.2522196, 0.3702323, -0.4277888, 0.5392530, -0.7914726, 0.7980211
8: -0.2879248, 0.3032581, -0.4597707, 0.4874637, -0.7753884, 0.7630287
9: -0.1518255, 0.2052653, -0.2950300, 0.3378728, -0.4896984, 0.5002953

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8768543, upper bound: 1.8721501
time: 2.13 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8782208, upper bound: 1.8781102
time: 2.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1166757, 0.6639405, -0.2756782, 1.0348582, -1.1515338, 0.9396188
1: -0.1867247, 0.2285658, -0.3691912, 0.3825276, -0.5692523, 0.5977571
2: -0.2726291, 0.2572156, -0.4555201, 0.4630998, -0.7357289, 0.7127357
3: -0.1932868, 0.1103589, -0.3186647, 0.2952608, -0.4885476, 0.4290236
4: -0.1692145, 0.2753257, -0.3729503, 0.4154344, -0.5846490, 0.6482759
5: -0.3275969, 0.3487661, -0.4941449, 0.5466849, -0.8742818, 0.8429110
6: 0.1520091, 1.2557375, -0.2578550, 1.3709097, -1.2189006, 1.5135925
7: -0.1903213, 0.3234246, -0.4031706, 0.5097649, -0.7000862, 0.7265952
8: -0.2385489, 0.2405939, -0.4231654, 0.4637560, -0.7023050, 0.6637593
9: -0.1116305, 0.1743712, -0.2779647, 0.3217283, -0.4333587, 0.4523359

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8760701, upper bound: 1.8735505
time: 2.24 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8765402, upper bound: 1.8756369
time: 2.73 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 6.53 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8858838, upper bound: 1.8840776
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8859925, upper bound: 1.8851126
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8852058, upper bound: 1.8840763
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8853412, upper bound: 1.8851109
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8826873, upper bound: 1.8727699
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8831230, upper bound: 1.8748591
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8819805, upper bound: 1.8727677
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8825025, upper bound: 1.8748561
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8962803, upper bound: 1.8964987
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8962803, upper bound: 1.8964987
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8956616, upper bound: 1.8965051
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8956616, upper bound: 1.8965051
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8922086, upper bound: 1.8865752
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8922086, upper bound: 1.8871205
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8917609, upper bound: 1.8865779
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8917609, upper bound: 1.8871236
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8964987, upper bound: 1.8962803
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8965051, upper bound: 1.8956616
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8967487, upper bound: 1.8969979
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8967487, upper bound: 1.8969979
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8946934, upper bound: 1.8870974
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8947441, upper bound: 1.8867529
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8943947, upper bound: 1.8877212
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8943947, upper bound: 1.8878554
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8961928, upper bound: 1.8966231
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8961952, upper bound: 1.8959426
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8964859, upper bound: 1.8972860
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8964859, upper bound: 1.8972860
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8936766, upper bound: 1.8877123
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8936795, upper bound: 1.8873748
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8933251, upper bound: 1.8882962
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8933251, upper bound: 1.8883937
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8629564, upper bound: 1.8603913
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8641842, upper bound: 1.8660429
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8657129, upper bound: 1.8605790
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8669637, upper bound: 1.8663260
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8629657, upper bound: 1.8596354
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8641842, upper bound: 1.8653880
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8657190, upper bound: 1.8598276
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8669649, upper bound: 1.8656382
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8771831, upper bound: 1.8721502
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8784567, upper bound: 1.8781099
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8768543, upper bound: 1.8721501
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8782208, upper bound: 1.8781102
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8760701, upper bound: 1.8735505
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.53
Output dim: 6, lower bound: -1.8765402, upper bound: 1.8756369
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8865121, upper bound: 1.8871076
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8868402, upper bound: 1.8870008
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8862572, upper bound: 1.8870034
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8873423, upper bound: 1.8870456
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8873423, upper bound: 1.8870456
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8874072, upper bound: 1.8877897
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8871057, upper bound: 1.8877924
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8880674, upper bound: 1.8877609
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8880674, upper bound: 1.8883800
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8411868, upper bound: 1.8103338
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8420023, upper bound: 1.8104184
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8414214, upper bound: 1.8103104
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8422578, upper bound: 1.8103867
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8576008, upper bound: 1.8223780
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8576008, upper bound: 1.8290884
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8568046, upper bound: 1.8275270
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8568046, upper bound: 1.8300619
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8522478, upper bound: 1.8204159
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8522478, upper bound: 1.8287593
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8550277, upper bound: 1.8271880
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8550277, upper bound: 1.8300520
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8286488, upper bound: 1.8412186
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8286488, upper bound: 1.8412186
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8297394, upper bound: 1.8430532
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8297394, upper bound: 1.8433551
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8297394, upper bound: 1.8430532
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.53
Output dim: 6, lower bound: -1.8297394, upper bound: 1.8433551
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.094357967376709
rel_dist={6: [-1.9113006693296792, 1.9113006693296786]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 189

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8708799, upper bound: 1.8605139
time: 2.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8604377, upper bound: 1.8604377
time: 3.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 6.52 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 6.52
Output dim: 6, lower bound: -1.8708799, upper bound: 1.8605139
IS_A2, status: Status.UNKNOWN, split count: 1, time: 6.52
Output dim: 6, lower bound: -1.8604377, upper bound: 1.8604377

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.4283954, 1.2709696, -0.4433304, 1.2907131, -1.7191085, 1.7143000
1: -0.5065504, 0.5120057, -0.5194843, 0.5232759, -1.0298263, 1.0314901
2: -0.5936686, 0.6343641, -0.6061395, 0.6519829, -1.2456515, 1.2405037
3: -0.4099944, 0.4626325, -0.4214497, 0.4766767, -0.8866711, 0.8840823
4: -0.5260350, 0.5839709, -0.5383946, 0.6068381, -1.1328731, 1.1223655
5: -0.6253715, 0.7425824, -0.6403219, 0.7615999, -1.3869714, 1.3829043
6: -0.5344308, 1.4685735, -0.5592921, 1.4756627, -2.0100935, 2.0278656
7: -0.5838713, 0.6439485, -0.6013403, 0.6563315, -1.2402029, 1.2452888
8: -0.5708863, 0.6576684, -0.5846704, 0.6780241, -1.2489104, 1.2423388
9: -0.4413340, 0.4900914, -0.4562192, 0.5076230, -0.9489570, 0.9463106

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8393140, upper bound: 1.8334733
time: 2.31 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8379985, upper bound: 1.8269292
time: 2.64 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.1578722, 2.0902753, -0.4017833, 1.2295885, -2.3874607, 2.4920588
1: -1.0186968, 0.9981120, -0.4813879, 0.4927728, -1.5114696, 1.4794998
2: -1.0924278, 1.3473158, -0.5699674, 0.6014031, -1.6938310, 1.9172832
3: -0.8712537, 1.0596817, -0.3895310, 0.4362993, -1.3075529, 1.4492127
4: -1.0514562, 1.4630030, -0.5015263, 0.5449609, -1.5964171, 1.9645293
5: -1.3404843, 1.4613965, -0.5974572, 0.7090652, -2.0495496, 2.0588536
6: -1.5363673, 1.8078189, -0.4843796, 1.4511552, -2.9875226, 2.2921987
7: -1.2797452, 1.1851227, -0.5526656, 0.6209165, -1.9006617, 1.7377883
8: -1.1523812, 1.4595408, -0.5440878, 0.6201960, -1.7725773, 2.0036287
9: -1.0169351, 1.1958456, -0.4139925, 0.4605213, -1.4774565, 1.6098380

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8268513, upper bound: 1.8325094
time: 3.50 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8268339, upper bound: 1.8268339
time: 2.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 7.46 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 7.46
Output dim: 6, lower bound: -1.8393140, upper bound: 1.8334733
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 7.46
Output dim: 6, lower bound: -1.8379985, upper bound: 1.8269292
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 7.46
Output dim: 6, lower bound: -1.8268513, upper bound: 1.8325094
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 7.46
Output dim: 6, lower bound: -1.8268339, upper bound: 1.8268339
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.094357967376709
rel_dist={6: [-1.9066430217684205, 1.9066430217684207]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 189

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8824145, upper bound: 1.8626747
time: 2.22 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8625167, upper bound: 1.8625167
time: 2.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 5.04 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 5.04
Output dim: 6, lower bound: -1.8824145, upper bound: 1.8626747
IS_A2, status: Status.UNKNOWN, split count: 1, time: 5.04
Output dim: 6, lower bound: -1.8625167, upper bound: 1.8625167

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.4283954, 1.2709696, -0.4578914, 1.3071958, -1.7355912, 1.7288611
1: -0.5065504, 0.5120057, -0.5303295, 0.5330924, -1.0396428, 1.0423352
2: -0.5936686, 0.6343641, -0.6167517, 0.6670228, -1.2606914, 1.2511158
3: -0.4099944, 0.4626325, -0.4311371, 0.4890101, -0.8990045, 0.8937697
4: -0.5260350, 0.5839709, -0.5489670, 0.6262327, -1.1522677, 1.1329379
5: -0.6253715, 0.7425824, -0.6545904, 0.7775620, -1.4029335, 1.3971727
6: -0.5344308, 1.4685735, -0.5801282, 1.4817449, -2.0161757, 2.0487018
7: -0.5838713, 0.6439485, -0.6160147, 0.6675788, -1.2514501, 1.2599633
8: -0.5708863, 0.6576684, -0.5962563, 0.6955847, -1.2664710, 1.2539246
9: -0.4413340, 0.4900914, -0.4686736, 0.5231478, -0.9644818, 0.9587650

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8527645, upper bound: 1.8405935
time: 2.63 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8505651, upper bound: 1.8288786
time: 2.96 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.1578722, 2.0902753, -0.4195073, 1.2561909, -2.4140630, 2.5097826
1: -1.0186968, 0.9981120, -0.4979559, 0.5060524, -1.5247493, 1.4960679
2: -1.0924278, 1.3473158, -0.5855699, 0.6239715, -1.7163993, 1.9328856
3: -0.8712537, 1.0596817, -0.4028323, 0.4543021, -1.3255558, 1.4625139
4: -1.0514562, 1.4630030, -0.5177632, 0.5713495, -1.6228057, 1.9807663
5: -1.3404843, 1.4613965, -0.6159214, 0.7322084, -2.0726926, 2.0773177
6: -1.5363673, 1.8078189, -0.5170974, 1.4613873, -2.9977546, 2.3249164
7: -1.2797452, 1.1851227, -0.5733691, 0.6361942, -1.9159393, 1.7584918
8: -1.1523812, 1.4595408, -0.5614452, 0.6460018, -1.7983830, 2.0209861
9: -1.0169351, 1.1958456, -0.4325337, 0.4810839, -1.4980190, 1.6283793

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8286990, upper bound: 1.8390184
time: 2.02 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8286669, upper bound: 1.8286669
time: 1.87 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.07 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.07
Output dim: 6, lower bound: -1.8527645, upper bound: 1.8405935
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.07
Output dim: 6, lower bound: -1.8505651, upper bound: 1.8288786
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 5.07
Output dim: 6, lower bound: -1.8286990, upper bound: 1.8390184
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 5.07
Output dim: 6, lower bound: -1.8286669, upper bound: 1.8286669

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.3358470, 1.1031274, -0.2533865, 0.9526966, -1.2885436, 1.3565139
1: -0.4158585, 0.4377563, -0.3379972, 0.3715008, -0.7873594, 0.7757536
2: -0.5026419, 0.5282035, -0.4210550, 0.4450719, -0.9477139, 0.9492585
3: -0.3478958, 0.3630051, -0.2970739, 0.2803935, -0.6282892, 0.6600790
4: -0.4293731, 0.4675858, -0.3406602, 0.4049166, -0.8342897, 0.8082460
5: -0.5309295, 0.6291055, -0.4582641, 0.5334337, -1.0643632, 1.0873696
6: -0.3440939, 1.3852478, -0.1777658, 1.3064377, -1.6505315, 1.5630137
7: -0.4731899, 0.5583558, -0.3826028, 0.4791976, -0.9523876, 0.9409586
8: -0.4683574, 0.5402008, -0.3905033, 0.4457758, -0.9141332, 0.9307041
9: -0.3453482, 0.3917409, -0.2679954, 0.3150626, -0.6604108, 0.6597364

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8486902, upper bound: 1.8346502
time: 2.22 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8527187, upper bound: 1.8405392
time: 2.21 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.2754635, 1.0003704, -0.2554146, 0.9460368, -1.2215003, 1.2557850
1: -0.3606810, 0.3868144, -0.3379242, 0.3745842, -0.7352651, 0.7247386
2: -0.4438091, 0.4664768, -0.4199460, 0.4485464, -0.8923554, 0.8864228
3: -0.3122305, 0.2992448, -0.2969564, 0.2837811, -0.5960116, 0.5962012
4: -0.3653402, 0.4197630, -0.3408460, 0.4082022, -0.7735424, 0.7606090
5: -0.4806853, 0.5553092, -0.4560604, 0.5392885, -1.0199738, 1.0113696
6: -0.2289478, 1.3358341, -0.1741105, 1.3030008, -1.5319486, 1.5099447
7: -0.4050156, 0.5023273, -0.3857206, 0.4791139, -0.8841295, 0.8880479
8: -0.4095523, 0.4685453, -0.3939690, 0.4502254, -0.8597777, 0.8625143
9: -0.2842758, 0.3318289, -0.2717072, 0.3207725, -0.6050483, 0.6035360

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8463266, upper bound: 1.8263452
time: 2.09 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8505159, upper bound: 1.8288786
time: 2.91 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.18 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.18
Output dim: 6, lower bound: -1.8486902, upper bound: 1.8346502
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.18
Output dim: 6, lower bound: -1.8527187, upper bound: 1.8405392
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.18
Output dim: 6, lower bound: -1.8463266, upper bound: 1.8263452
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.18
Output dim: 6, lower bound: -1.8505159, upper bound: 1.8288786

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3413104, 1.1305139, -0.1933479, 0.8777977, -1.2191081, 1.3238618
1: -0.4249827, 0.4397967, -0.2894593, 0.3229269, -0.7479097, 0.7292560
2: -0.5136102, 0.5318394, -0.3760882, 0.3821949, -0.8958052, 0.9079276
3: -0.3543903, 0.3670664, -0.2643922, 0.2220153, -0.5764055, 0.6314585
4: -0.4393592, 0.4695189, -0.2833960, 0.3588699, -0.7982291, 0.7529149
5: -0.5419399, 0.6290777, -0.4196543, 0.4645492, -1.0064890, 1.0487320
6: -0.3691089, 1.4078717, -0.0849700, 1.2819319, -1.6510408, 1.4928417
7: -0.4786476, 0.5671106, -0.3188597, 0.4262916, -0.9049392, 0.8859703
8: -0.4813717, 0.5429657, -0.3380977, 0.3776336, -0.8590053, 0.8810634
9: -0.3487714, 0.3916594, -0.2105301, 0.2530450, -0.6018164, 0.6021894

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 189

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8486902, upper bound: 1.8346502
time: 2.90 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8486902, upper bound: 1.8346502
time: 2.13 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3074237, 1.0577087, -0.2435330, 0.9402428, -1.2476665, 1.3012418
1: -0.3900880, 0.4119457, -0.3299664, 0.3635747, -0.7536627, 0.7419121
2: -0.4746600, 0.4984653, -0.4136155, 0.4347898, -0.9094498, 0.9120808
3: -0.3322071, 0.3287601, -0.2916939, 0.2708067, -0.6030138, 0.6204540
4: -0.3988611, 0.4431736, -0.3311912, 0.3974410, -0.7963021, 0.7743648
5: -0.5085462, 0.5911726, -0.4517689, 0.5223226, -1.0308688, 1.0429415
6: -0.2927718, 1.3688600, -0.1624719, 1.3021684, -1.5949402, 1.5313319
7: -0.4387682, 0.5333434, -0.3721337, 0.4704770, -0.9092453, 0.9054771
8: -0.4422156, 0.5043337, -0.3820396, 0.4346988, -0.8769143, 0.8863733
9: -0.3134121, 0.3604664, -0.2585224, 0.3049614, -0.6183735, 0.6189888

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 189

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8527187, upper bound: 1.8405392
time: 2.59 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8527187, upper bound: 1.8405392
time: 2.83 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2885560, 1.0404780, -0.1961107, 0.8720319, -1.1605879, 1.2365886
1: -0.3768210, 0.3940753, -0.2898388, 0.3266325, -0.7034535, 0.6839141
2: -0.4612783, 0.4775532, -0.3752810, 0.3864650, -0.8477433, 0.8528342
3: -0.3235258, 0.3092770, -0.2644713, 0.2260823, -0.5496082, 0.5737483
4: -0.3830085, 0.4261124, -0.2841252, 0.3627797, -0.7457881, 0.7102376
5: -0.4984969, 0.5627595, -0.4175230, 0.4714238, -0.9699207, 0.9802824
6: -0.2683901, 1.3659037, -0.0826546, 1.2839736, -1.5523636, 1.4485583
7: -0.4175678, 0.5186957, -0.3227324, 0.4267608, -0.8443286, 0.8414282
8: -0.4297578, 0.4791240, -0.3423309, 0.3829817, -0.8127395, 0.8214549
9: -0.2930151, 0.3378691, -0.2148555, 0.2595191, -0.5525342, 0.5527246

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 189

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8463266, upper bound: 1.8263452
time: 1.94 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8463266, upper bound: 1.8263452
time: 2.31 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2470492, 0.9641068, -0.2453630, 0.9332970, -1.1803463, 1.2094698
1: -0.3375776, 0.3639501, -0.3297060, 0.3664982, -0.7040758, 0.6936561
2: -0.4224954, 0.4367438, -0.4123391, 0.4380400, -0.8605354, 0.8490829
3: -0.2967498, 0.2715551, -0.2914377, 0.2740053, -0.5707551, 0.5629928
4: -0.3380548, 0.3981804, -0.3311691, 0.4005685, -0.7386233, 0.7293495
5: -0.4617109, 0.5232152, -0.4492603, 0.5279323, -0.9896432, 0.9724755
6: -0.1845749, 1.3211735, -0.1584880, 1.2994087, -1.4839835, 1.4796616
7: -0.3747494, 0.4772831, -0.3750371, 0.4701976, -0.8449471, 0.8523202
8: -0.3850949, 0.4365317, -0.3853135, 0.4389007, -0.8239956, 0.8218452
9: -0.2569091, 0.3026542, -0.2620459, 0.3104683, -0.5673773, 0.5647001

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 189

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8505159, upper bound: 1.8288786
time: 2.84 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8505159, upper bound: 1.8288786
time: 2.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 6.71 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.71
Output dim: 6, lower bound: -1.8486902, upper bound: 1.8346502
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.71
Output dim: 6, lower bound: -1.8486902, upper bound: 1.8346502
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.71
Output dim: 6, lower bound: -1.8527187, upper bound: 1.8405392
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.71
Output dim: 6, lower bound: -1.8527187, upper bound: 1.8405392
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.71
Output dim: 6, lower bound: -1.8463266, upper bound: 1.8263452
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.71
Output dim: 6, lower bound: -1.8463266, upper bound: 1.8263452
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.71
Output dim: 6, lower bound: -1.8505159, upper bound: 1.8288786
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.71
Output dim: 6, lower bound: -1.8505159, upper bound: 1.8288786

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3413104, 1.1305139, -0.1740291, 0.8539119, -1.1952223, 1.3045430
1: -0.4249827, 0.4397967, -0.2738202, 0.3072972, -0.7322799, 0.7136168
2: -0.5136102, 0.5318394, -0.3614711, 0.3620415, -0.8756518, 0.8933105
3: -0.3543903, 0.3670664, -0.2537802, 0.2032469, -0.5576372, 0.6208466
4: -0.4393592, 0.4695189, -0.2651047, 0.3437925, -0.7831517, 0.7346236
5: -0.5419399, 0.6290777, -0.4074909, 0.4417679, -0.9837077, 1.0365686
6: -0.3691089, 1.4078717, -0.0547634, 1.2745473, -1.6436563, 1.4626352
7: -0.4786476, 0.5671106, -0.2982344, 0.4094291, -0.8880768, 0.8653450
8: -0.4813717, 0.5429657, -0.3207724, 0.3553792, -0.8367509, 0.8637382
9: -0.3487714, 0.3916594, -0.1923214, 0.2328746, -0.5816460, 0.5839808

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8398567, upper bound: 1.8199588
time: 2.11 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8486902, upper bound: 1.8346497
time: 2.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3413104, 1.1305139, -0.5858635, 1.4298468, -1.7711571, 1.7163774
1: -0.4249827, 0.4397967, -0.6195089, 0.6293713, -1.0543540, 1.0593055
2: -0.5136102, 0.5318394, -0.6854938, 0.7831785, -1.2967887, 1.2173331
3: -0.3543903, 0.3670664, -0.4850665, 0.5985661, -0.9529564, 0.8521329
4: -0.4393592, 0.4695189, -0.6680137, 0.6496086, -1.0889678, 1.1375326
5: -0.5419399, 0.6290777, -0.7092387, 0.8922794, -1.4342194, 1.3383164
6: -0.3691089, 1.4078717, -0.7383192, 1.5553229, -1.9244318, 2.1461909
7: -0.4786476, 0.5671106, -0.7350265, 0.7780809, -1.2567285, 1.3021371
8: -0.4813717, 0.5429657, -0.6857212, 0.8117003, -1.2930720, 1.2286868
9: -0.3487714, 0.3916594, -0.5785924, 0.6417103, -0.9904817, 0.9702518

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8398567, upper bound: 1.8199588
time: 1.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8486902, upper bound: 1.8346497
time: 2.16 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3074237, 1.0577087, -0.2239318, 0.9152642, -1.2226880, 1.2816405
1: -0.3900880, 0.4119457, -0.3140576, 0.3476228, -0.7377107, 0.7260033
2: -0.4746600, 0.4984653, -0.3987921, 0.4141876, -0.8888475, 0.8972574
3: -0.3322071, 0.3287601, -0.2808622, 0.2517689, -0.5839760, 0.6096224
4: -0.3988611, 0.4431736, -0.3124415, 0.3822175, -0.7810786, 0.7556151
5: -0.5085462, 0.5911726, -0.4388354, 0.4993283, -1.0078745, 1.0300080
6: -0.2927718, 1.3688600, -0.1319848, 1.2929424, -1.5857142, 1.5008447
7: -0.4387682, 0.5333434, -0.3512907, 0.4531863, -0.8919545, 0.8846341
8: -0.4422156, 0.5043337, -0.3645227, 0.4122363, -0.8544519, 0.8688563
9: -0.3134121, 0.3604664, -0.2399317, 0.2845931, -0.5980052, 0.6003981

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8471447, upper bound: 1.8302848
time: 2.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8527129, upper bound: 1.8405273
time: 2.52 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3074237, 1.0577087, -0.6379259, 1.4951291, -1.8025528, 1.6956346
1: -0.3900880, 0.4119457, -0.6609594, 0.6739317, -1.0640197, 1.0729051
2: -0.4746600, 0.4984653, -0.7238286, 0.8419824, -1.3166424, 1.2222939
3: -0.3322071, 0.3287601, -0.5128599, 0.6497301, -0.9819372, 0.8416200
4: -0.3988611, 0.4431736, -0.7197316, 0.6889124, -1.0877736, 1.1629052
5: -0.5085462, 0.5911726, -0.7460322, 0.9512165, -1.4597627, 1.3372048
6: -0.2927718, 1.3688600, -0.8169401, 1.5895399, -1.8823117, 2.1858001
7: -0.4387682, 0.5333434, -0.7892787, 0.8264967, -1.2652650, 1.3226221
8: -0.4422156, 0.5043337, -0.7292591, 0.8697459, -1.3119614, 1.2335927
9: -0.3134121, 0.3604664, -0.6312523, 0.6945597, -1.0079718, 0.9917188

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8471447, upper bound: 1.8302848
time: 2.42 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8527129, upper bound: 1.8405273
time: 2.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2885560, 1.0404780, -0.1769939, 0.8478454, -1.1364014, 1.2174718
1: -0.3768210, 0.3940753, -0.2742930, 0.3111618, -0.6879828, 0.6683683
2: -0.4612783, 0.4775532, -0.3607699, 0.3665335, -0.8278117, 0.8383232
3: -0.3235258, 0.3092770, -0.2538303, 0.2075185, -0.5310444, 0.5631074
4: -0.3830085, 0.4261124, -0.2659830, 0.3478879, -0.7308964, 0.6920954
5: -0.4984969, 0.5627595, -0.4052576, 0.4489160, -0.9474130, 0.9680170
6: -0.2683901, 1.3659037, -0.0527614, 1.2775522, -1.5459423, 1.4186652
7: -0.4175678, 0.5186957, -0.3023337, 0.4099706, -0.8275385, 0.8210294
8: -0.4297578, 0.4791240, -0.3251902, 0.3610308, -0.7907887, 0.8043141
9: -0.2930151, 0.3378691, -0.1968622, 0.2396016, -0.5326167, 0.5347313

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8369762, upper bound: 1.8135940
time: 5.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8463266, upper bound: 1.8263452
time: 2.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2885560, 1.0404780, -0.5929876, 1.4357104, -1.7242665, 1.6334655
1: -0.3768210, 0.3940753, -0.6242278, 0.6380554, -1.0148764, 1.0183030
2: -0.4612783, 0.4775532, -0.6896216, 0.7954426, -1.2567208, 1.1671748
3: -0.3235258, 0.3092770, -0.4881979, 0.6059853, -0.9295112, 0.7974750
4: -0.3830085, 0.4261124, -0.6757441, 0.6555938, -1.0386022, 1.1018565
5: -0.4984969, 0.5627595, -0.7135617, 0.9024011, -1.4008980, 1.2763212
6: -0.2683901, 1.3659037, -0.7465609, 1.5590847, -1.8274748, 2.1124647
7: -0.4175678, 0.5186957, -0.7416578, 0.7864477, -1.2040155, 1.2603536
8: -0.4297578, 0.4791240, -0.6903651, 0.8201274, -1.2498852, 1.1694890
9: -0.2930151, 0.3378691, -0.5872934, 0.6490441, -0.9420592, 0.9251625

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8369762, upper bound: 1.8135940
time: 2.26 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8463266, upper bound: 1.8263452
time: 2.48 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2470492, 0.9641068, -0.2259986, 0.9087049, -1.1557541, 1.1901054
1: -0.3375776, 0.3639501, -0.3139102, 0.3507490, -0.6883266, 0.6778603
2: -0.4224954, 0.4367438, -0.3975760, 0.4177181, -0.8402135, 0.8343198
3: -0.2967498, 0.2715551, -0.2806406, 0.2552162, -0.5519660, 0.5521957
4: -0.3380548, 0.3981804, -0.3126049, 0.3855376, -0.7235924, 0.7107853
5: -0.4617109, 0.5232152, -0.4363339, 0.5052395, -0.9669504, 0.9595492
6: -0.1845749, 1.3211735, -0.1284308, 1.2917348, -1.4763098, 1.4496044
7: -0.3747494, 0.4772831, -0.3544739, 0.4530713, -0.8278207, 0.8317570
8: -0.3850949, 0.4365317, -0.3680155, 0.4167238, -0.8018187, 0.8045472
9: -0.2569091, 0.3026542, -0.2437109, 0.2903667, -0.5472758, 0.5463651

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8448273, upper bound: 1.8235094
time: 2.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8505154, upper bound: 1.8288786
time: 2.63 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2470492, 0.9641068, -0.6429430, 1.4995207, -1.7465699, 1.6070497
1: -0.3375776, 0.3639501, -0.6649954, 0.6785982, -1.0161757, 1.0289454
2: -0.4224954, 0.4367438, -0.7272417, 0.8479278, -1.2704232, 1.1639855
3: -0.2967498, 0.2715551, -0.5155492, 0.6547326, -0.9514824, 0.7871043
4: -0.3380548, 0.3981804, -0.7238128, 0.6940682, -1.0321230, 1.1219932
5: -0.4617109, 0.5232152, -0.7495351, 0.9600410, -1.4217519, 1.2727504
6: -0.1845749, 1.3211735, -0.8235905, 1.5910509, -1.7756257, 2.1447639
7: -0.3747494, 0.4772831, -0.7948948, 0.8307832, -1.2055327, 1.2721779
8: -0.3850949, 0.4365317, -0.7327535, 0.8769477, -1.2620425, 1.1692852
9: -0.2569091, 0.3026542, -0.6352550, 0.7009045, -0.9578136, 0.9379092

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8448273, upper bound: 1.8235094
time: 2.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8505154, upper bound: 1.8288786
time: 2.88 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.46 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.46
Output dim: 6, lower bound: -1.8398567, upper bound: 1.8199588
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 6, lower bound: -1.8486902, upper bound: 1.8346497
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.46
Output dim: 6, lower bound: -1.8398567, upper bound: 1.8199588
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 6, lower bound: -1.8486902, upper bound: 1.8346497
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 6, lower bound: -1.8471447, upper bound: 1.8302848
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 6, lower bound: -1.8527129, upper bound: 1.8405273
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 6, lower bound: -1.8471447, upper bound: 1.8302848
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 6, lower bound: -1.8527129, upper bound: 1.8405273
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.46
Output dim: 6, lower bound: -1.8369762, upper bound: 1.8135940
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 6, lower bound: -1.8463266, upper bound: 1.8263452
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.46
Output dim: 6, lower bound: -1.8369762, upper bound: 1.8135940
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 6, lower bound: -1.8463266, upper bound: 1.8263452
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 6, lower bound: -1.8448273, upper bound: 1.8235094
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 6, lower bound: -1.8505154, upper bound: 1.8288786
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 6, lower bound: -1.8448273, upper bound: 1.8235094
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 6, lower bound: -1.8505154, upper bound: 1.8288786

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3160911, 1.0913444, -0.1670138, 0.8457954, -1.1618865, 1.2583582
1: -0.4024585, 0.4166000, -0.2682254, 0.3016567, -0.7041152, 0.6848254
2: -0.4892096, 0.5053321, -0.3563249, 0.3547479, -0.8439574, 0.8616569
3: -0.3408500, 0.3361698, -0.2500463, 0.1963507, -0.5372007, 0.5862162
4: -0.4125052, 0.4468142, -0.2585016, 0.3384342, -0.7509394, 0.7053157
5: -0.5226758, 0.5952024, -0.4034123, 0.4338277, -0.9565035, 0.9986148
6: -0.3244798, 1.3946694, -0.0441412, 1.2728379, -1.5973177, 1.4388106
7: -0.4476634, 0.5453157, -0.2907093, 0.4033943, -0.8510576, 0.8360250
8: -0.4584269, 0.5111086, -0.3147853, 0.3474433, -0.8058702, 0.8258939
9: -0.3196910, 0.3635175, -0.1854774, 0.2255394, -0.5452304, 0.5489949

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8855914, upper bound: 1.8902670
time: 2.48 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8856054, upper bound: 1.8898911
time: 3.22 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3160911, 1.0913444, -0.5788783, 1.4212675, -1.7373586, 1.6702226
1: -0.4024585, 0.4166000, -0.6138644, 0.6237450, -1.0262035, 1.0304644
2: -0.4892096, 0.5053321, -0.6803415, 0.7759132, -1.2651229, 1.1856735
3: -0.3408500, 0.3361698, -0.4813423, 0.5916964, -0.9325464, 0.8175122
4: -0.4125052, 0.4468142, -0.6613555, 0.6442624, -1.0567677, 1.1081697
5: -0.5226758, 0.5952024, -0.7043146, 0.8843646, -1.4070404, 1.2995172
6: -0.3244798, 1.3946694, -0.7278088, 1.5513008, -1.8757806, 2.1224782
7: -0.4476634, 0.5453157, -0.7275369, 0.7720560, -1.2197194, 1.2728525
8: -0.4584269, 0.5111086, -0.6799148, 0.8037878, -1.2622147, 1.1910233
9: -0.3196910, 0.3635175, -0.5717657, 0.6343910, -0.9540820, 0.9352832

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8478917, upper bound: 1.8345827
time: 2.45 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8484801, upper bound: 1.8345291
time: 2.46 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2903790, 1.0641489, -0.1716743, 0.8522161, -1.1425951, 1.2358232
1: -0.3838948, 0.3933299, -0.2721720, 0.3052350, -0.6891298, 0.6655020
2: -0.4703371, 0.4778280, -0.3600817, 0.3594140, -0.8297511, 0.8379097
3: -0.3286973, 0.3082746, -0.2526501, 0.2006313, -0.5293286, 0.5609248
4: -0.3891063, 0.4260251, -0.2630661, 0.3418486, -0.7309549, 0.6890912
5: -0.5080880, 0.5629197, -0.4065796, 0.4388404, -0.9469284, 0.9694993
6: -0.2903615, 1.3916360, -0.0520106, 1.2745037, -1.5648652, 1.4436467
7: -0.4183930, 0.5248778, -0.2954432, 0.4076762, -0.8260691, 0.8203211
8: -0.4399841, 0.4798564, -0.3187060, 0.3525881, -0.7925722, 0.7985624
9: -0.2893438, 0.3343385, -0.1895898, 0.2299337, -0.5192775, 0.5239283

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8862824, upper bound: 1.8908452
time: 2.50 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8862976, upper bound: 1.8904915
time: 2.19 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2813155, 1.0241079, -0.2168672, 0.9067461, -1.1880617, 1.2409751
1: -0.3691451, 0.3896427, -0.3084235, 0.3419018, -0.7110469, 0.6980662
2: -0.4535598, 0.4710432, -0.3936270, 0.4067735, -0.8603333, 0.8646703
3: -0.3182806, 0.3027263, -0.2770888, 0.2448240, -0.5631046, 0.5798151
4: -0.3738989, 0.4226248, -0.3057358, 0.3768422, -0.7507410, 0.7283606
5: -0.4909417, 0.5593879, -0.4344764, 0.4913696, -0.9823113, 0.9938644
6: -0.2514104, 1.3560926, -0.1213360, 1.2905282, -1.5419385, 1.4774286
7: -0.4101071, 0.5107104, -0.3437460, 0.4470185, -0.8571256, 0.8544564
8: -0.4204467, 0.4733283, -0.3585107, 0.4042730, -0.8247197, 0.8318390
9: -0.2859444, 0.3331180, -0.2329909, 0.2772350, -0.5631794, 0.5661088

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8865577, upper bound: 1.8924795
time: 3.14 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8865734, upper bound: 1.8918976
time: 2.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2903790, 1.0641489, -0.5853978, 1.4289598, -1.7193389, 1.6495466
1: -0.3838948, 0.3933299, -0.6183096, 0.6311297, -1.0150245, 1.0116396
2: -0.4703371, 0.4778280, -0.6845588, 0.7865966, -1.2569337, 1.1623869
3: -0.3286973, 0.3082746, -0.4843266, 0.5982593, -0.9269565, 0.7926012
4: -0.3891063, 0.4260251, -0.6694045, 0.6482373, -1.0373436, 1.0954295
5: -0.5080880, 0.5629197, -0.7084528, 0.8902271, -1.3983152, 1.2713726
6: -0.2903615, 1.3916360, -0.7367864, 1.5571116, -1.8474731, 2.1284223
7: -0.4183930, 0.5248778, -0.7330965, 0.7801845, -1.1985774, 1.2579744
8: -0.4399841, 0.4798564, -0.6850324, 0.8097647, -1.2497489, 1.1648889
9: -0.2893438, 0.3343385, -0.5805409, 0.6396283, -0.9289721, 0.9148794

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 168

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8463397, upper bound: 1.8302225
time: 2.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8469835, upper bound: 1.8302021
time: 2.84 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2813155, 1.0241079, -0.6308841, 1.4864461, -1.7677617, 1.6549920
1: -0.3691451, 0.3896427, -0.6552581, 0.6682210, -1.0373662, 1.0449009
2: -0.4535598, 0.4710432, -0.7186464, 0.8345801, -1.2881398, 1.1896896
3: -0.3182806, 0.3027263, -0.5090889, 0.6427953, -0.9610758, 0.8118151
4: -0.3738989, 0.4226248, -0.7129639, 0.6835409, -1.0574398, 1.1355886
5: -0.4909417, 0.5593879, -0.7410772, 0.9432603, -1.4342020, 1.3004651
6: -0.2514104, 1.3560926, -0.8063571, 1.5854605, -1.8368709, 2.1624498
7: -0.4101071, 0.5107104, -0.7817459, 0.8203021, -1.2304091, 1.2924562
8: -0.4204467, 0.4733283, -0.7234113, 0.8617983, -1.2822449, 1.1967396
9: -0.2859444, 0.3331180, -0.6243244, 0.6872066, -0.9731510, 0.9574423

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 168

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8520381, upper bound: 1.8404731
time: 2.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8526764, upper bound: 1.8404430
time: 2.50 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2637203, 1.0101281, -0.1699842, 0.8396559, -1.1033762, 1.1801124
1: -0.3571510, 0.3739540, -0.2686767, 0.3055276, -0.6626787, 0.6426307
2: -0.4432597, 0.4514457, -0.3556141, 0.3592511, -0.8025107, 0.8070599
3: -0.3103930, 0.2848232, -0.2500635, 0.2006307, -0.5110238, 0.5348867
4: -0.3595317, 0.4072330, -0.2593663, 0.3425355, -0.7020671, 0.6665993
5: -0.4824580, 0.5346835, -0.4011374, 0.4409850, -0.9234430, 0.9358209
6: -0.2310573, 1.3536762, -0.0421415, 1.2761897, -1.5072470, 1.3958178
7: -0.3910259, 0.4972295, -0.2948171, 0.4038873, -0.7949132, 0.7920466
8: -0.4090845, 0.4511936, -0.3192081, 0.3531053, -0.7621897, 0.7704017
9: -0.2685857, 0.3120100, -0.1900299, 0.2322780, -0.5008637, 0.5020399

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8747115, upper bound: 1.8727056
time: 2.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8751318, upper bound: 1.8755367
time: 2.32 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2637203, 1.0101281, -0.5860103, 1.4271282, -1.6908485, 1.5961385
1: -0.3571510, 0.3739540, -0.6185835, 0.6323909, -0.9895419, 0.9925375
2: -0.4432597, 0.4514457, -0.6844906, 0.7881068, -1.2313665, 1.1359364
3: -0.3103930, 0.2848232, -0.4844590, 0.5991097, -0.9095027, 0.7692822
4: -0.3595317, 0.4072330, -0.6690404, 0.6502698, -1.0098015, 1.0762734
5: -0.4824580, 0.5346835, -0.7086555, 0.8945120, -1.3769699, 1.2433391
6: -0.2310573, 1.3536762, -0.7360802, 1.5550617, -1.7861190, 2.0897565
7: -0.3910259, 0.4972295, -0.7341927, 0.7803199, -1.1713457, 1.2314222
8: -0.4090845, 0.4511936, -0.6845720, 0.8122551, -1.2213396, 1.1357656
9: -0.2685857, 0.3120100, -0.5804223, 0.6417533, -0.9103390, 0.8924323

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8453824, upper bound: 1.8262921
time: 2.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8461357, upper bound: 1.8262921
time: 2.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2332312, 0.9775003, -0.1744263, 0.8455284, -1.0787597, 1.1519266
1: -0.3341753, 0.3489960, -0.2723966, 0.3089215, -0.6430968, 0.6213925
2: -0.4227687, 0.4191557, -0.3591030, 0.3636782, -0.7864469, 0.7782587
3: -0.2951883, 0.2541907, -0.2524918, 0.2047635, -0.4999518, 0.5066825
4: -0.3315992, 0.3840640, -0.2636899, 0.3457375, -0.6773367, 0.6477538
5: -0.4657531, 0.5007770, -0.4040511, 0.4456007, -0.9113539, 0.9048281
6: -0.1885641, 1.3442752, -0.0494128, 1.2767978, -1.4653618, 1.3936881
7: -0.3581440, 0.4718401, -0.2993678, 0.4079419, -0.7660859, 0.7712078
8: -0.3853513, 0.4169149, -0.3227957, 0.3579220, -0.7432733, 0.7397106
9: -0.2371952, 0.2793809, -0.1940719, 0.2364850, -0.4736803, 0.4734527

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8862480, upper bound: 1.8861491
time: 2.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8862626, upper bound: 1.8858009
time: 4.01 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2209890, 0.9326779, -0.2189445, 0.9002141, -1.1212031, 1.1516223
1: -0.3169004, 0.3428235, -0.3082526, 0.3450369, -0.6619373, 0.6510761
2: -0.4035277, 0.4093708, -0.3923621, 0.4103202, -0.8138478, 0.8017329
3: -0.2829252, 0.2459040, -0.2768409, 0.2482799, -0.5312052, 0.5227449
4: -0.3134107, 0.3783336, -0.3058736, 0.3801702, -0.6935809, 0.6842072
5: -0.4456532, 0.4938234, -0.4319586, 0.4972918, -0.9429450, 0.9257821
6: -0.1453341, 1.3101858, -0.1178015, 1.2899866, -1.4353206, 1.4279873
7: -0.3468922, 0.4546568, -0.3469371, 0.4468963, -0.7937885, 0.8015939
8: -0.3630796, 0.4071459, -0.3620076, 0.4087756, -0.7718552, 0.7691535
9: -0.2312621, 0.2754708, -0.2367875, 0.2830203, -0.5142824, 0.5122583

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8757703, upper bound: 1.8734067
time: 2.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8762761, upper bound: 1.8762761
time: 2.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2332312, 0.9775003, -0.5916669, 1.4347217, -1.6679529, 1.5691673
1: -0.3341753, 0.3489960, -0.6232063, 0.6368404, -0.9710157, 0.9722022
2: -0.4227687, 0.4191557, -0.6888200, 0.7938717, -1.2166404, 1.1079757
3: -0.2951883, 0.2541907, -0.4875190, 0.6045363, -0.8997246, 0.7417097
4: -0.3315992, 0.3840640, -0.6745430, 0.6543940, -0.9859933, 1.0586070
5: -0.4657531, 0.5007770, -0.7127944, 0.9004888, -1.3662419, 1.2135714
6: -0.1885641, 1.3442752, -0.7450644, 1.5592433, -1.7478074, 2.0893397
7: -0.3581440, 0.4718401, -0.7400951, 0.7853647, -1.1435087, 1.2119353
8: -0.3853513, 0.4169149, -0.6894830, 0.8184179, -1.2037692, 1.1063979
9: -0.2371952, 0.2793809, -0.5858448, 0.6473472, -0.8845424, 0.8652256

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 122

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8439323, upper bound: 1.8234332
time: 3.47 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8447548, upper bound: 1.8234332
time: 3.00 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2209890, 0.9326779, -0.6359544, 1.4909172, -1.7119062, 1.5686324
1: -0.3169004, 0.3428235, -0.6593301, 0.6729288, -0.9898292, 1.0021536
2: -0.4035277, 0.4093708, -0.7220818, 0.8405790, -1.2441067, 1.1314526
3: -0.2829252, 0.2459040, -0.5118022, 0.6478484, -0.9307736, 0.7577062
4: -0.3134107, 0.3783336, -0.7170895, 0.6887400, -1.0021507, 1.0954232
5: -0.4456532, 0.4938234, -0.7446220, 0.9521473, -1.3978006, 1.2384454
6: -0.1453341, 1.3101858, -0.8130636, 1.5870306, -1.7323647, 2.1232495
7: -0.3468922, 0.4546568, -0.7874174, 0.8246213, -1.1715136, 1.2420743
8: -0.3630796, 0.4071459, -0.7269495, 0.8690599, -1.2321396, 1.1340953
9: -0.2312621, 0.2754708, -0.6283712, 0.6936056, -0.9248677, 0.9038420

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 168

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8497176, upper bound: 1.8288717
time: 2.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8504747, upper bound: 1.8288717
time: 2.52 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.44 seconds
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8855914, upper bound: 1.8902670
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8856054, upper bound: 1.8898911
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8478917, upper bound: 1.8345827
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8484801, upper bound: 1.8345291
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8862824, upper bound: 1.8908452
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8862976, upper bound: 1.8904915
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8865577, upper bound: 1.8924795
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8865734, upper bound: 1.8918976
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8463397, upper bound: 1.8302225
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8469835, upper bound: 1.8302021
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8520381, upper bound: 1.8404731
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8526764, upper bound: 1.8404430
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8747115, upper bound: 1.8727056
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8751318, upper bound: 1.8755367
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8453824, upper bound: 1.8262921
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8461357, upper bound: 1.8262921
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8862480, upper bound: 1.8861491
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8862626, upper bound: 1.8858009
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8757703, upper bound: 1.8734067
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8762761, upper bound: 1.8762761
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8439323, upper bound: 1.8234332
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8447548, upper bound: 1.8234332
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8497176, upper bound: 1.8288717
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.44
Output dim: 6, lower bound: -1.8504747, upper bound: 1.8288717

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2839905, 1.0461637, -0.1125035, 0.6615767, -0.9455673, 1.1586671
1: -0.3758053, 0.3892478, -0.1851427, 0.2255192, -0.6013245, 0.5743905
2: -0.4618428, 0.4717896, -0.2711246, 0.2509224, -0.7127652, 0.7429142
3: -0.3233194, 0.3035270, -0.1914360, 0.1078557, -0.4311751, 0.4949629
4: -0.3809766, 0.4216374, -0.1672613, 0.2713735, -0.6523501, 0.5888987
5: -0.4997721, 0.5557482, -0.3273860, 0.3415149, -0.8412870, 0.8831342
6: -0.2708502, 1.3753887, 0.1571152, 1.2467669, -1.5176171, 1.2182736
7: -0.4121518, 0.5170740, -0.1856136, 0.3221811, -0.7343329, 0.7026876
8: -0.4304976, 0.4729206, -0.2329060, 0.2339069, -0.6644046, 0.7058266
9: -0.2863278, 0.3303003, -0.1085948, 0.1694204, -0.4557482, 0.4388951

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8712365, upper bound: 1.8785137
time: 3.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8772162, upper bound: 1.8816570
time: 2.29 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2957828, 1.0614448, -0.1337333, 0.7951134, -1.0908961, 1.1951780
1: -0.3853431, 0.3987209, -0.2383201, 0.2724366, -0.6577797, 0.6370410
2: -0.4706602, 0.4841365, -0.3288391, 0.3195179, -0.7901781, 0.8129755
3: -0.3296897, 0.3149941, -0.2319908, 0.1551452, -0.4848349, 0.5469849
4: -0.3922507, 0.4305656, -0.2274971, 0.3130493, -0.7053000, 0.6580628
5: -0.5079011, 0.5691083, -0.3795029, 0.3973026, -0.9052037, 0.9486112
6: -0.2893084, 1.3823154, 0.0180405, 1.2633083, -1.5526167, 1.3642750
7: -0.4246769, 0.5274177, -0.2524728, 0.3732100, -0.7978868, 0.7798905
8: -0.4405995, 0.4861703, -0.2867428, 0.3020450, -0.7426445, 0.7729131
9: -0.2976546, 0.3423582, -0.1515615, 0.2030560, -0.5007105, 0.4939197

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8721033, upper bound: 1.8785501
time: 2.49 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8742019, upper bound: 1.8793614
time: 2.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2839905, 1.0461637, -0.4284664, 1.2135034, -1.4974939, 1.4746301
1: -0.3758053, 0.3892478, -0.4887260, 0.5051863, -0.8809916, 0.8779737
2: -0.4618428, 0.4717896, -0.5632956, 0.6214392, -1.0832820, 1.0350852
3: -0.3233194, 0.3035270, -0.3975213, 0.4466438, -0.7699633, 0.7010483
4: -0.3809766, 0.4216374, -0.5150883, 0.5315840, -0.9125606, 0.9367256
5: -0.4997721, 0.5557482, -0.5928050, 0.7175890, -1.2173610, 1.1485533
6: -0.2708502, 1.3753887, -0.4823093, 1.4385053, -1.7093555, 1.8576981
7: -0.4121518, 0.5170740, -0.5677894, 0.6386831, -1.0508349, 1.0848634
8: -0.4304976, 0.4729206, -0.5461050, 0.6360573, -1.0665549, 1.0190256
9: -0.2863278, 0.3303003, -0.4298099, 0.4831145, -0.7694423, 0.7601101

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8315201, upper bound: 1.8193749
time: 2.15 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8362860, upper bound: 1.8226287
time: 2.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2957828, 1.0614448, -0.5352150, 1.3651855, -1.6609683, 1.5966598
1: -0.3853431, 0.3987209, -0.5783733, 0.5888956, -0.9742388, 0.9770943
2: -0.4706602, 0.4841365, -0.6475874, 0.7308098, -1.2014700, 1.1317239
3: -0.3296897, 0.3149941, -0.4577828, 0.5492064, -0.8788961, 0.7727769
4: -0.3922507, 0.4305656, -0.6196781, 0.6111724, -1.0034232, 1.0502437
5: -0.5079011, 0.5691083, -0.6732386, 0.8353700, -1.3432710, 1.2423470
6: -0.2893084, 1.3823154, -0.6602186, 1.5229604, -1.8122689, 2.0425339
7: -0.4246769, 0.5274177, -0.6810880, 0.7340982, -1.1587751, 1.2085056
8: -0.4405995, 0.4861703, -0.6427867, 0.7546687, -1.1952682, 1.1289570
9: -0.2976546, 0.3423582, -0.5298085, 0.5894814, -0.8871359, 0.8721666

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8352360, upper bound: 1.8219472
time: 2.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8362394, upper bound: 1.8222531
time: 2.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2553120, 1.0188355, -0.1126660, 0.6676732, -0.9229852, 1.1315016
1: -0.3555521, 0.3652461, -0.1876048, 0.2270288, -0.5825809, 0.5528509
2: -0.4440280, 0.4412388, -0.2737800, 0.2540432, -0.6980712, 0.7150188
3: -0.3097640, 0.2742144, -0.1933468, 0.1092467, -0.4190108, 0.4675612
4: -0.3555849, 0.3995966, -0.1700463, 0.2730435, -0.6286284, 0.5696430
5: -0.4852008, 0.5236160, -0.3297020, 0.3439725, -0.8291733, 0.8533179
6: -0.2351314, 1.3708079, 0.1507005, 1.2474608, -1.4825922, 1.2201074
7: -0.3812158, 0.4940835, -0.1881590, 0.3246985, -0.7059143, 0.6822425
8: -0.4098884, 0.4405931, -0.2352963, 0.2369953, -0.6468837, 0.6758894
9: -0.2556740, 0.2986263, -0.1087675, 0.1708345, -0.4265085, 0.4073937

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8723553, upper bound: 1.8790154
time: 2.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8779013, upper bound: 1.8822694
time: 3.03 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2671140, 1.0342025, -0.1370096, 0.8013018, -1.0684159, 1.1712120
1: -0.3651391, 0.3746836, -0.2416334, 0.2754035, -0.6405426, 0.6163169
2: -0.4529440, 0.4535509, -0.3319216, 0.3230665, -0.7760105, 0.7854725
3: -0.3161625, 0.2856457, -0.2339917, 0.1590869, -0.4752493, 0.5196375
4: -0.3668815, 0.4085012, -0.2309234, 0.3155968, -0.6824783, 0.6394246
5: -0.4928543, 0.5368997, -0.3822201, 0.4010629, -0.8939172, 0.9191198
6: -0.2539099, 1.3780203, 0.0108343, 1.2643833, -1.5182931, 1.3671860
7: -0.3937143, 0.5044614, -0.2564146, 0.3764705, -0.7701848, 0.7608761
8: -0.4200839, 0.4538230, -0.2896335, 0.3059207, -0.7260045, 0.7434565
9: -0.2669176, 0.3106054, -0.1542706, 0.2050995, -0.4720171, 0.4648760

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8723568, upper bound: 1.8785816
time: 3.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8779047, upper bound: 1.8818210
time: 16.41 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2468315, 0.9798963, -0.1166893, 0.7168427, -0.9636742, 1.0965856
1: -0.3412496, 0.3619430, -0.2061583, 0.2409284, -0.5821780, 0.5681013
2: -0.4277136, 0.4350762, -0.2937946, 0.2832608, -0.7109744, 0.7288707
3: -0.2996695, 0.2692154, -0.2091215, 0.1215587, -0.4212283, 0.4783369
4: -0.3408665, 0.3965461, -0.1919994, 0.2884338, -0.6293002, 0.5885455
5: -0.4681846, 0.5207131, -0.3471230, 0.3675942, -0.8357787, 0.8678361
6: -0.1977003, 1.3370545, 0.1009418, 1.2580063, -1.4557066, 1.2361126
7: -0.3735446, 0.4803142, -0.2118796, 0.3431585, -0.7167031, 0.6921938
8: -0.3908849, 0.4346208, -0.2572018, 0.2654126, -0.6562974, 0.6918226
9: -0.2527714, 0.2978758, -0.1240962, 0.1855257, -0.4382971, 0.4219720

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8730656, upper bound: 1.8812000
time: 3.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8758554, upper bound: 1.8819838
time: 3.07 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2582109, 0.9946620, -0.1733909, 0.8528180, -1.1110289, 1.1680529
1: -0.3504796, 0.3710818, -0.2731687, 0.3071514, -0.6576310, 0.6442505
2: -0.4363050, 0.4469514, -0.3609307, 0.3616844, -0.7979894, 0.8078821
3: -0.3058397, 0.2802468, -0.2534247, 0.2026013, -0.5084411, 0.5336715
4: -0.3517659, 0.4051701, -0.2642073, 0.3439964, -0.6957623, 0.6693774
5: -0.4756272, 0.5335436, -0.4072109, 0.4427385, -0.9183657, 0.9407545
6: -0.2155776, 1.3434466, -0.0536224, 1.2773002, -1.4928778, 1.3970690
7: -0.3856018, 0.4903722, -0.2975863, 0.4086696, -0.7942714, 0.7879585
8: -0.4007006, 0.4474300, -0.3214891, 0.3555088, -0.7562094, 0.7689191
9: -0.2636489, 0.3094840, -0.1912576, 0.2328236, -0.4964726, 0.5007417

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8730718, upper bound: 1.8806896
time: 2.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8758614, upper bound: 1.8813842
time: 2.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2553120, 1.0188355, -0.4349236, 1.2216078, -1.4769198, 1.4537592
1: -0.3555521, 0.3652461, -0.4934709, 0.5116274, -0.8671795, 0.8587170
2: -0.4440280, 0.4412388, -0.5678437, 0.6304129, -1.0744410, 1.0090824
3: -0.3097640, 0.2742144, -0.4007124, 0.4529592, -0.7627233, 0.6749268
4: -0.3555849, 0.3995966, -0.5222996, 0.5358287, -0.8914136, 0.9218963
5: -0.4852008, 0.5236160, -0.5971506, 0.7238412, -1.2090421, 1.1207665
6: -0.2351314, 1.3708079, -0.4917954, 1.4440464, -1.6791778, 1.8626033
7: -0.3812158, 0.4940835, -0.5737127, 0.6458707, -1.0270865, 1.0677962
8: -0.4098884, 0.4405931, -0.5514303, 0.6424024, -1.0522908, 0.9920234
9: -0.2556740, 0.2986263, -0.4374032, 0.4887023, -0.7443763, 0.7360295

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8296325, upper bound: 1.8143882
time: 2.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8345283, upper bound: 1.8180018
time: 2.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2671140, 1.0342025, -0.5415490, 1.3728025, -1.6399165, 1.5757514
1: -0.3651391, 0.3746836, -0.5827355, 0.5958654, -0.9610045, 0.9574190
2: -0.4529440, 0.4535509, -0.6517743, 0.7408213, -1.1937654, 1.1053252
3: -0.3161625, 0.2856457, -0.4607345, 0.5555360, -0.8716985, 0.7463802
4: -0.3668815, 0.4085012, -0.6271657, 0.6151024, -0.9819839, 1.0356669
5: -0.4928543, 0.5368997, -0.6772854, 0.8411787, -1.3340330, 1.2141851
6: -0.2539099, 1.3780203, -0.6690570, 1.5285609, -1.7824707, 2.0470772
7: -0.3937143, 0.5044614, -0.6865790, 0.7417203, -1.1354346, 1.1910404
8: -0.4200839, 0.4538230, -0.6477982, 0.7605695, -1.1806533, 1.1016212
9: -0.2669176, 0.3106054, -0.5380716, 0.5946534, -0.8615710, 0.8486770

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8304323, upper bound: 1.8143702
time: 2.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8352373, upper bound: 1.8180017
time: 2.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2468315, 0.9798963, -0.4803246, 1.2784119, -1.5252434, 1.4602208
1: -0.3412496, 0.3619430, -0.5300000, 0.5485581, -0.8898078, 0.8919430
2: -0.4277136, 0.4350762, -0.6015332, 0.6782842, -1.1059978, 1.0366094
3: -0.2996695, 0.2692154, -0.4252238, 0.4973730, -0.7970425, 0.6944392
4: -0.3408665, 0.3965461, -0.5654420, 0.5709180, -0.9117845, 0.9619880
5: -0.4681846, 0.5207131, -0.6290837, 0.7766141, -1.2447987, 1.1497967
6: -0.1977003, 1.3370545, -0.5610121, 1.4721547, -1.6698551, 1.8980666
7: -0.3735446, 0.4803142, -0.6221460, 0.6854774, -1.0590221, 1.1024601
8: -0.3908849, 0.4346208, -0.5896607, 0.6941767, -1.0850616, 1.0242815
9: -0.2527714, 0.2978758, -0.4810944, 0.5360672, -0.7888386, 0.7789702

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8383943, upper bound: 1.8277196
time: 2.07 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8402777, upper bound: 1.8287064
time: 2.10 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2582109, 0.9946620, -0.5873719, 1.4303088, -1.6885197, 1.5820340
1: -0.3504796, 0.3710818, -0.6198127, 0.6332132, -0.9836928, 0.9908945
2: -0.4363050, 0.4469514, -0.6859652, 0.7891165, -1.2254214, 1.1329167
3: -0.3058397, 0.2802468, -0.4855201, 0.6003518, -0.9061916, 0.7657669
4: -0.3517659, 0.4051701, -0.6709902, 0.6506214, -1.0023873, 1.0761603
5: -0.4756272, 0.5335436, -0.7097015, 0.8945159, -1.3701431, 1.2432451
6: -0.2155776, 1.3434466, -0.7389495, 1.5570641, -1.7726417, 2.0823960
7: -0.3856018, 0.4903722, -0.7354953, 0.7818732, -1.1674750, 1.2258675
8: -0.4007006, 0.4474300, -0.6863487, 0.8129610, -1.2136617, 1.1337786
9: -0.2636489, 0.3094840, -0.5821669, 0.6425331, -0.9061820, 0.8916509

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8390270, upper bound: 1.8277078
time: 3.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8409174, upper bound: 1.8286061
time: 2.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1678486, 0.8724900, -0.1165565, 0.6034849, -0.7713335, 0.9890465
1: -0.2754107, 0.2982641, -0.1581975, 0.2126162, -0.4880269, 0.4564615
2: -0.3655934, 0.3521460, -0.2444077, 0.2254607, -0.5910541, 0.5965537
3: -0.2551445, 0.1921506, -0.1710630, 0.0989075, -0.3540519, 0.3632137
4: -0.2644722, 0.3368840, -0.1398350, 0.2584420, -0.5229142, 0.4767190
5: -0.4153891, 0.4306356, -0.3030027, 0.3216115, -0.7370006, 0.7336383
6: -0.0651370, 1.2888702, 0.2183098, 1.2525871, -1.3177241, 1.0705605
7: -0.2895210, 0.4089923, -0.1658056, 0.2947397, -0.5842607, 0.5747978
8: -0.3196794, 0.3438161, -0.2150107, 0.2101669, -0.5298463, 0.5588268
9: -0.1787779, 0.2202858, -0.1107063, 0.1625452, -0.3413231, 0.3309921

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8746471, upper bound: 1.8722892
time: 2.80 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8739224, upper bound: 1.8722912
time: 2.78 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1839426, 0.9004468, -0.1185251, 0.6762528, -0.8601955, 1.0189718
1: -0.2903126, 0.3113454, -0.1913821, 0.2326753, -0.5229880, 0.5027275
2: -0.3800581, 0.3694439, -0.2774998, 0.2672504, -0.6473085, 0.6469437
3: -0.2654905, 0.2084671, -0.1977392, 0.1139300, -0.3794205, 0.4062063
4: -0.2814508, 0.3481781, -0.1751308, 0.2802484, -0.5616993, 0.5233088
5: -0.4282384, 0.4476016, -0.3313746, 0.3563910, -0.7846295, 0.7789761
6: -0.0980127, 1.3020376, 0.1387196, 1.2597359, -1.3577487, 1.1633179
7: -0.3068747, 0.4251972, -0.1980424, 0.3278307, -0.6347054, 0.6232396
8: -0.3366337, 0.3630488, -0.2453210, 0.2500961, -0.5867297, 0.6083698
9: -0.1938949, 0.2333881, -0.1144525, 0.1796590, -0.3735539, 0.3478406

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8750382, upper bound: 1.8748296
time: 2.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8741803, upper bound: 1.8748328
time: 3.10 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2315099, 0.9687929, -0.4458828, 1.2331660, -1.4646759, 1.4146757
1: -0.3311246, 0.3481131, -0.5022192, 0.5210634, -0.8521880, 0.8503323
2: -0.4190431, 0.4177866, -0.5753958, 0.6427131, -1.0617561, 0.9931824
3: -0.2929247, 0.2534786, -0.4063471, 0.4638657, -0.7567904, 0.6598257
4: -0.3287674, 0.3829083, -0.5319665, 0.5454593, -0.8742266, 0.9148748
5: -0.4617596, 0.4984072, -0.6044788, 0.7394250, -1.2011846, 1.1028860
6: -0.1804693, 1.3354366, -0.5077789, 1.4494249, -1.6298943, 1.8432155
7: -0.3567848, 0.4689512, -0.5857631, 0.6549187, -1.0117035, 1.0547143
8: -0.3813201, 0.4151016, -0.5599258, 0.6563007, -1.0376208, 0.9750274
9: -0.2376284, 0.2791352, -0.4473199, 0.5012589, -0.7388873, 0.7264550

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8320296, upper bound: 1.8138904
time: 2.97 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8330055, upper bound: 1.8143476
time: 2.30 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2433687, 0.9837739, -0.5419124, 1.3703216, -1.6136903, 1.5256863
1: -0.3406837, 0.3576453, -0.5827078, 0.5968956, -0.9375793, 0.9403530
2: -0.4279266, 0.4302009, -0.6513560, 0.7420188, -1.1699455, 1.0815569
3: -0.2993393, 0.2650316, -0.4605285, 0.5560808, -0.8554201, 0.7255601
4: -0.3400528, 0.3918910, -0.6265131, 0.6169006, -0.9569533, 1.0184041
5: -0.4692534, 0.5118479, -0.6770862, 0.8450836, -1.3143370, 1.1889341
6: -0.1990446, 1.3419769, -0.6678162, 1.5263966, -1.7254412, 2.0097930
7: -0.3694065, 0.4793171, -0.6873035, 0.7414310, -1.1108375, 1.1666206
8: -0.3915148, 0.4284179, -0.6469828, 0.7627576, -1.1542723, 1.0754007
9: -0.2490207, 0.2912740, -0.5376710, 0.5964664, -0.8454870, 0.8289449

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8327335, upper bound: 1.8138904
time: 2.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8338420, upper bound: 1.8143458
time: 2.27 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1981094, 0.9339187, -0.1164064, 0.6728417, -0.8709511, 1.0503250
1: -0.3058050, 0.3209592, -0.1900225, 0.2306240, -0.5364290, 0.5109817
2: -0.3963607, 0.3826676, -0.2761504, 0.2622570, -0.6586176, 0.6588180
3: -0.2761724, 0.2201107, -0.1959928, 0.1122245, -0.3883969, 0.4161035
4: -0.2982558, 0.3575784, -0.1731911, 0.2775966, -0.5758524, 0.5307695
5: -0.4438434, 0.4613575, -0.3306129, 0.3518703, -0.7957137, 0.7919704
6: -0.1332214, 1.3248705, 0.1431792, 1.2552507, -1.3884721, 1.1816913
7: -0.3208652, 0.4411905, -0.1942875, 0.3267702, -0.6476354, 0.6354780
8: -0.3551536, 0.3775696, -0.2416750, 0.2451709, -0.6003245, 0.6192446
9: -0.2035978, 0.2436034, -0.1115584, 0.1763728, -0.3799705, 0.3551617

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8723310, upper bound: 1.8759968
time: 2.89 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8778676, upper bound: 1.8776889
time: 2.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2100156, 0.9487565, -0.1377625, 0.7935474, -1.0035630, 1.0865190
1: -0.3154504, 0.3304060, -0.2405622, 0.2781333, -0.5935838, 0.5709682
2: -0.4053777, 0.3949541, -0.3298024, 0.3256852, -0.7310629, 0.7247565
3: -0.2826491, 0.2316154, -0.2327664, 0.1625184, -0.4451675, 0.4643818
4: -0.3094603, 0.3665781, -0.2296648, 0.3182268, -0.6276870, 0.5962430
5: -0.4511845, 0.4747905, -0.3787017, 0.4060112, -0.8571957, 0.8534921
6: -0.1521727, 1.3316233, 0.0148965, 1.2685248, -1.4206975, 1.3167269
7: -0.3334973, 0.4515604, -0.2590327, 0.3750361, -0.7085334, 0.7105931
8: -0.3654483, 0.3909460, -0.2921942, 0.3094371, -0.6748854, 0.6831402
9: -0.2148485, 0.2557090, -0.1568222, 0.2085870, -0.4234355, 0.4125313

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8723334, upper bound: 1.8755846
time: 3.19 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8778732, upper bound: 1.8774154
time: 3.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1332252, 0.7951291, -0.1208419, 0.6408699, -0.7740951, 0.9159710
1: -0.2379069, 0.2710322, -0.1780573, 0.2259663, -0.4638732, 0.4490894
2: -0.3290138, 0.3181818, -0.2630478, 0.2532342, -0.5822480, 0.5812296
3: -0.2320544, 0.1519383, -0.1880057, 0.1069949, -0.3390493, 0.3399441
4: -0.2266662, 0.3130245, -0.1604457, 0.2730920, -0.4997582, 0.4734702
5: -0.3802408, 0.3984268, -0.3174468, 0.3463150, -0.7265558, 0.7158735
6: 0.0185675, 1.2699380, 0.1732713, 1.2624515, -1.2438841, 1.0966667
7: -0.2499930, 0.3732447, -0.1855422, 0.3145140, -0.5645070, 0.5587869
8: -0.2881189, 0.3015466, -0.2346191, 0.2368353, -0.5249542, 0.5361657
9: -0.1489050, 0.2031746, -0.1141350, 0.1744746, -0.3233796, 0.3173095

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8757170, upper bound: 1.8730435
time: 2.91 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8754232, upper bound: 1.8730520
time: 2.31 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1449192, 0.8203365, -0.1230558, 0.7303898, -0.8753091, 0.9433923
1: -0.2505742, 0.2816933, -0.2119019, 0.2566078, -0.5071820, 0.4935952
2: -0.3410411, 0.3313149, -0.3014582, 0.2984617, -0.6395028, 0.6327732
3: -0.2399051, 0.1663129, -0.2149903, 0.1333649, -0.3732699, 0.3813032
4: -0.2398153, 0.3221777, -0.1993802, 0.3006771, -0.5404924, 0.5215579
5: -0.3911667, 0.4120734, -0.3520733, 0.3813492, -0.7725158, 0.7641467
6: -0.0100458, 1.2748332, 0.0826570, 1.2700298, -1.2800756, 1.1921762
7: -0.2649270, 0.3850418, -0.2276692, 0.3485767, -0.6135038, 0.6127110
8: -0.2990249, 0.3157113, -0.2706238, 0.2811162, -0.5801412, 0.5863351
9: -0.1585060, 0.2107719, -0.1376121, 0.1951403, -0.3536463, 0.3483840

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8761974, upper bound: 1.8758338
time: 3.19 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8758409, upper bound: 1.8758409
time: 2.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1981094, 0.9339187, -0.4521230, 1.2413218, -1.4394312, 1.3860416
1: -0.3058050, 0.3209592, -0.5072021, 0.5260001, -0.8318052, 0.8281613
2: -0.3963607, 0.3826676, -0.5801224, 0.6490661, -1.0454267, 0.9627900
3: -0.2761724, 0.2201107, -0.4097063, 0.4698379, -0.7460103, 0.6298170
4: -0.2982558, 0.3575784, -0.5379623, 0.5500244, -0.8482802, 0.8955407
5: -0.4438434, 0.4613575, -0.6089734, 0.7460393, -1.1898826, 1.0703309
6: -0.1332214, 1.3248705, -0.5173839, 1.4538473, -1.5870687, 1.8422544
7: -0.3208652, 0.4411905, -0.5922211, 0.6604302, -0.9812953, 1.0334117
8: -0.3551536, 0.3775696, -0.5652919, 0.6631075, -1.0182611, 0.9428614
9: -0.2035978, 0.2436034, -0.4533105, 0.5074401, -0.7110379, 0.6969139

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8274761, upper bound: 1.8086146
time: 2.27 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8319603, upper bound: 1.8111851
time: 2.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2100156, 0.9487565, -0.5476032, 1.3777928, -1.5878085, 1.4963597
1: -0.3154504, 0.3304060, -0.5873021, 0.6013687, -0.9168192, 0.9177080
2: -0.4053777, 0.3949541, -0.6556735, 0.7478051, -1.1531829, 1.0506276
3: -0.2826491, 0.2316154, -0.4635786, 0.5615125, -0.8441616, 0.6951941
4: -0.3094603, 0.3665781, -0.6320254, 0.6210278, -0.9304881, 0.9986035
5: -0.4511845, 0.4747905, -0.6811917, 0.8510400, -1.3022245, 1.1559823
6: -0.1521727, 1.3316233, -0.6766871, 1.5305498, -1.6827224, 2.0083103
7: -0.3334973, 0.4515604, -0.6931875, 0.7464560, -1.0799533, 1.1447480
8: -0.3654483, 0.3909460, -0.6518761, 0.7689459, -1.1343943, 1.0428221
9: -0.2148485, 0.2557090, -0.5431406, 0.6020682, -0.8169168, 0.7988496

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8282805, upper bound: 1.8086146
time: 2.17 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8327980, upper bound: 1.8111851
time: 2.52 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1865402, 0.8896887, -0.4962321, 1.2969812, -1.4835215, 1.3859208
1: -0.2889638, 0.3152601, -0.5428124, 0.5619389, -0.8509027, 0.8580725
2: -0.3775719, 0.3735987, -0.6131753, 0.6955810, -1.0731529, 0.9867740
3: -0.2642377, 0.2124140, -0.4337825, 0.5129787, -0.7772164, 0.6461965
4: -0.2805324, 0.3522397, -0.5799553, 0.5842265, -0.8647590, 0.9321950
5: -0.4239965, 0.4551005, -0.6401801, 0.7974843, -1.2214808, 1.0952806
6: -0.0913250, 1.2947183, -0.5850586, 1.4814906, -1.5728157, 1.8797768
7: -0.3102399, 0.4244052, -0.6393670, 0.6989958, -1.0092357, 1.0637722
8: -0.3334148, 0.3684336, -0.6026455, 0.7135495, -1.0469643, 0.9710791
9: -0.1982540, 0.2402212, -0.4956709, 0.5535235, -0.7517775, 0.7358921

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8361150, upper bound: 1.8167618
time: 2.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8378456, upper bound: 1.8170390
time: 1.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1979861, 0.9039791, -0.5922395, 1.4342754, -1.6322615, 1.4962187
1: -0.2982955, 0.3243856, -0.6235780, 0.6377553, -0.9360508, 0.9479635
2: -0.3862655, 0.3854290, -0.6891607, 0.7948797, -1.1811452, 1.0745897
3: -0.2704864, 0.2235207, -0.4880235, 0.6051921, -0.8756784, 0.7115441
4: -0.2914463, 0.3609430, -0.6747655, 0.6556583, -0.9471046, 1.0357085
5: -0.4310995, 0.4680581, -0.7130898, 0.9031301, -1.3342296, 1.1811479
6: -0.1094243, 1.2998818, -0.7450655, 1.5585641, -1.6679883, 2.0449471
7: -0.3224418, 0.4344198, -0.7409095, 0.7858014, -1.1082432, 1.1753293
8: -0.3433388, 0.3813441, -0.6896475, 0.8199775, -1.1633162, 1.0709916
9: -0.2091215, 0.2519408, -0.5860119, 0.6487188, -0.8578403, 0.8379527

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8367783, upper bound: 1.8167618
time: 2.43 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8385417, upper bound: 1.8170390
time: 2.58 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.34 seconds
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8712365, upper bound: 1.8785137
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8772162, upper bound: 1.8816570
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8721033, upper bound: 1.8785501
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8742019, upper bound: 1.8793614
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8315201, upper bound: 1.8193749
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8362860, upper bound: 1.8226287
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8352360, upper bound: 1.8219472
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8362394, upper bound: 1.8222531
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8723553, upper bound: 1.8790154
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8779013, upper bound: 1.8822694
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8723568, upper bound: 1.8785816
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8779047, upper bound: 1.8818210
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8730656, upper bound: 1.8812000
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8758554, upper bound: 1.8819838
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8730718, upper bound: 1.8806896
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8758614, upper bound: 1.8813842
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8296325, upper bound: 1.8143882
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8345283, upper bound: 1.8180018
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8304323, upper bound: 1.8143702
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8352373, upper bound: 1.8180017
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8383943, upper bound: 1.8277196
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8402777, upper bound: 1.8287064
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8390270, upper bound: 1.8277078
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8409174, upper bound: 1.8286061
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8746471, upper bound: 1.8722892
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8739224, upper bound: 1.8722912
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8750382, upper bound: 1.8748296
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8741803, upper bound: 1.8748328
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8320296, upper bound: 1.8138904
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8330055, upper bound: 1.8143476
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8327335, upper bound: 1.8138904
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8338420, upper bound: 1.8143458
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8723310, upper bound: 1.8759968
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8778676, upper bound: 1.8776889
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8723334, upper bound: 1.8755846
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8778732, upper bound: 1.8774154
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8757170, upper bound: 1.8730435
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8754232, upper bound: 1.8730520
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8761974, upper bound: 1.8758338
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8758409, upper bound: 1.8758409
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8274761, upper bound: 1.8086146
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8319603, upper bound: 1.8111851
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8282805, upper bound: 1.8086146
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8327980, upper bound: 1.8111851
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8361150, upper bound: 1.8167618
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8378456, upper bound: 1.8170390
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8367783, upper bound: 1.8167618
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.34
Output dim: 6, lower bound: -1.8385417, upper bound: 1.8170390

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1086159, 0.6190966, -0.1077906, 0.5234365, -0.6320524, 0.7268872
1: -0.1614838, 0.2113799, -0.1143235, 0.1751449, -0.3366287, 0.3257034
2: -0.2488376, 0.2235537, -0.2011289, 0.2106812, -0.4595188, 0.4246826
3: -0.1728996, 0.0986440, -0.1278926, 0.0802354, -0.2531350, 0.2265366
4: -0.1445486, 0.2559819, -0.1031728, 0.2190357, -0.3635843, 0.3591547
5: -0.3097528, 0.3151914, -0.2653082, 0.2583163, -0.5680692, 0.5804996
6: 0.2100367, 1.2353567, 0.3376632, 1.2338862, -1.0238495, 0.8976935
7: -0.1649648, 0.2988392, -0.1217934, 0.2403628, -0.4053276, 0.4206326
8: -0.2116387, 0.2073943, -0.1653995, 0.1964357, -0.4080743, 0.3727938
9: -0.1049192, 0.1578112, -0.1024259, 0.1356876, -0.2406069, 0.2602371

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8588657, upper bound: 1.8647502
time: 2.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8590809, upper bound: 1.8666937
time: 2.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1345758, 0.8090149, -0.1101195, 0.5839983, -0.7185740, 0.9191344
1: -0.2419349, 0.2700861, -0.1412968, 0.2003536, -0.4422885, 0.4113829
2: -0.3336432, 0.3176422, -0.2296301, 0.2123048, -0.5459481, 0.5472724
3: -0.2349055, 0.1517382, -0.1560822, 0.0933682, -0.3282737, 0.3078204
4: -0.2311380, 0.3113263, -0.1266339, 0.2447511, -0.4758891, 0.4379602
5: -0.3862196, 0.3934996, -0.2946188, 0.2963079, -0.6825275, 0.6881184
6: 0.0084141, 1.2639447, 0.2527220, 1.2373514, -1.2289373, 1.0112227
7: -0.2503902, 0.3774250, -0.1498479, 0.2778099, -0.5282001, 0.5272729
8: -0.2855876, 0.2997704, -0.1971043, 0.1984245, -0.4840121, 0.4968747
9: -0.1489110, 0.1998848, -0.1054499, 0.1515359, -0.3004469, 0.3053347

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8647601, upper bound: 1.8682885
time: 3.04 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8650325, upper bound: 1.8702528
time: 2.98 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1321697, 0.8081160, -0.1132798, 0.6654732, -0.7976429, 0.9213958
1: -0.2402219, 0.2658479, -0.1865785, 0.2264873, -0.4667093, 0.4524263
2: -0.3326158, 0.3131615, -0.2726467, 0.2527379, -0.5853537, 0.5858082
3: -0.2342922, 0.1454946, -0.1926801, 0.1090147, -0.3433069, 0.3381747
4: -0.2285730, 0.3088247, -0.1686351, 0.2726231, -0.5011961, 0.4774598
5: -0.3864612, 0.3903596, -0.3288470, 0.3438281, -0.7302893, 0.7192066
6: 0.0120834, 1.2666839, 0.1526753, 1.2487425, -1.2366590, 1.1140087
7: -0.2449219, 0.3765743, -0.1875227, 0.3232945, -0.5682163, 0.5640970
8: -0.2848982, 0.2951357, -0.2349599, 0.2358383, -0.5207365, 0.5300956
9: -0.1445175, 0.1970357, -0.1091702, 0.1707310, -0.3152485, 0.3062059

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8612718, upper bound: 1.8639292
time: 2.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8626877, upper bound: 1.8688729
time: 2.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1818000, 0.8976220, -0.1136689, 0.6848948, -0.8666948, 1.0112909
1: -0.2883992, 0.3097571, -0.1941752, 0.2310163, -0.5194155, 0.5039323
2: -0.3783061, 0.3671634, -0.2808132, 0.2624348, -0.6407409, 0.6479766
3: -0.2643217, 0.2064985, -0.1986604, 0.1133727, -0.3776944, 0.4051589
4: -0.2794189, 0.3465636, -0.1771986, 0.2776691, -0.5570880, 0.5237622
5: -0.4269809, 0.4450772, -0.3362328, 0.3513139, -0.7782948, 0.7813100
6: -0.0937811, 1.3005095, 0.1327847, 1.2506992, -1.3444803, 1.1677248
7: -0.3045745, 0.4232811, -0.1956468, 0.3308637, -0.6354381, 0.6189280
8: -0.3346959, 0.3605137, -0.2423023, 0.2451884, -0.5798844, 0.6028160
9: -0.1921212, 0.2313475, -0.1108264, 0.1751050, -0.3672263, 0.3421739

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8635809, upper bound: 1.8648069
time: 2.25 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8650357, upper bound: 1.8696955
time: 2.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1078762, 0.6060405, -0.1078960, 0.5245762, -0.6324524, 0.7139364
1: -0.1498499, 0.2030364, -0.1145554, 0.1751999, -0.3250498, 0.3175918
2: -0.2390071, 0.2086188, -0.2014207, 0.2108059, -0.4498129, 0.4100395
3: -0.1618150, 0.0962919, -0.1280996, 0.0804304, -0.2422453, 0.2243915
4: -0.1358402, 0.2471391, -0.1034379, 0.2191001, -0.3549403, 0.3505770
5: -0.3053637, 0.2995128, -0.2657025, 0.2585908, -0.5639545, 0.5652153
6: 0.2325973, 1.2340938, 0.3368877, 1.2341468, -1.0015495, 0.8972061
7: -0.1533656, 0.2866835, -0.1218649, 0.2407421, -0.3941077, 0.4085484
8: -0.2010968, 0.1949237, -0.1656746, 0.1965362, -0.3976330, 0.3605983
9: -0.1039380, 0.1514384, -0.1024986, 0.1356827, -0.2396207, 0.2539370

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8601372, upper bound: 1.8654675
time: 2.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8604160, upper bound: 1.8674452
time: 2.48 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1228521, 0.7780713, -0.1102682, 0.5868680, -0.7097201, 0.8883395
1: -0.2294863, 0.2479112, -0.1429293, 0.2013141, -0.4308004, 0.3908404
2: -0.3193447, 0.2949786, -0.2312762, 0.2124715, -0.5318162, 0.5262548
3: -0.2251650, 0.1307591, -0.1573259, 0.0939601, -0.3191251, 0.2880850
4: -0.2154533, 0.2956826, -0.1281269, 0.2457537, -0.4612070, 0.4238094
5: -0.3747460, 0.3772982, -0.2958565, 0.2979691, -0.6727151, 0.6731547
6: 0.0429090, 1.2620696, 0.2491440, 1.2376606, -1.1947517, 1.0129256
7: -0.2254765, 0.3658875, -0.1510496, 0.2795759, -0.5050523, 0.5169371
8: -0.2702950, 0.2777188, -0.1984687, 0.1985848, -0.4688798, 0.4761874
9: -0.1283356, 0.1884438, -0.1055907, 0.1521604, -0.2804960, 0.2940345

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8659501, upper bound: 1.8689603
time: 2.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8663704, upper bound: 1.8710082
time: 2.53 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1085063, 0.6123288, -0.1109791, 0.5845172, -0.6930236, 0.7233078
1: -0.1543445, 0.2059977, -0.1442562, 0.2033065, -0.3576510, 0.3502539
2: -0.2431619, 0.2130411, -0.2320080, 0.2137005, -0.4568624, 0.4450490
3: -0.1656963, 0.0976131, -0.1594592, 0.0939059, -0.2596022, 0.2570723
4: -0.1395236, 0.2502736, -0.1283080, 0.2479486, -0.3874722, 0.3785816
5: -0.3080365, 0.3050840, -0.2940216, 0.3021109, -0.6101474, 0.5991056
6: 0.2231461, 1.2358284, 0.2471825, 1.2392865, -1.0161405, 0.9886459
7: -0.1572972, 0.2913261, -0.1537970, 0.2812883, -0.4385855, 0.4451231
8: -0.2051613, 0.1988772, -0.2007514, 0.1998249, -0.4049862, 0.3996286
9: -0.1045569, 0.1536283, -0.1061416, 0.1542806, -0.2588376, 0.2597699

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8601392, upper bound: 1.8649158
time: 2.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8604171, upper bound: 1.8668605
time: 2.28 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1270012, 0.7911692, -0.1134883, 0.6770145, -0.8040157, 0.9046575
1: -0.2344072, 0.2546829, -0.1909858, 0.2298338, -0.4642409, 0.4456687
2: -0.3254524, 0.3022312, -0.2773390, 0.2610450, -0.5864974, 0.5795702
3: -0.2292243, 0.1357662, -0.1965176, 0.1118466, -0.3410709, 0.3322838
4: -0.2212575, 0.3009128, -0.1742050, 0.2763825, -0.4976400, 0.4751178
5: -0.3799446, 0.3829451, -0.3323486, 0.3491880, -0.7291326, 0.7152936
6: 0.0288506, 1.2653104, 0.1405827, 1.2490834, -1.2202327, 1.1247277
7: -0.2330012, 0.3709337, -0.1937400, 0.3278889, -0.5608901, 0.5646737
8: -0.2769023, 0.2848849, -0.2400498, 0.2434435, -0.5203458, 0.5249347
9: -0.1341564, 0.1918938, -0.1105709, 0.1743843, -0.3085408, 0.3024647

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8659638, upper bound: 1.8686813
time: 2.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8663737, upper bound: 1.8704570
time: 3.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1163663, 0.7245546, -0.1141166, 0.6080820, -0.7244483, 0.8386711
1: -0.2096887, 0.2397116, -0.1576864, 0.2112192, -0.4209079, 0.3973979
2: -0.2977187, 0.2774386, -0.2449102, 0.2225879, -0.5203066, 0.5223488
3: -0.2103753, 0.1220625, -0.1700267, 0.0987864, -0.3091617, 0.2920892
4: -0.1936803, 0.2873396, -0.1405632, 0.2565277, -0.4502080, 0.4279028
5: -0.3530967, 0.3668000, -0.3052665, 0.3174620, -0.6705587, 0.6720665
6: 0.0938960, 1.2609404, 0.2183297, 1.2472188, -1.1533229, 1.0426108
7: -0.2093243, 0.3462935, -0.1638446, 0.2947675, -0.5040917, 0.5101380
8: -0.2571072, 0.2611281, -0.2127358, 0.2075142, -0.4646214, 0.4738638
9: -0.1181753, 0.1829951, -0.1088944, 0.1601517, -0.2783270, 0.2918896

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 226

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8433373, upper bound: 1.8599637
time: 2.45 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8430796, upper bound: 1.8510437
time: 2.25 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1382200, 0.8106883, -0.1144625, 0.6186848, -0.7569047, 0.9251509
1: -0.2444066, 0.2745922, -0.1645530, 0.2151434, -0.4595500, 0.4391452
2: -0.3358434, 0.3230688, -0.2511621, 0.2292878, -0.5651312, 0.5742308
3: -0.2364387, 0.1564827, -0.1754526, 0.1007361, -0.3371747, 0.3319353
4: -0.2334495, 0.3162147, -0.1464006, 0.2606208, -0.4940702, 0.4626153
5: -0.3874206, 0.4030584, -0.3099925, 0.3245122, -0.7119327, 0.7130508
6: 0.0027168, 1.2730895, 0.2044734, 1.2486384, -1.2459216, 1.0686162
7: -0.2554380, 0.3793525, -0.1690523, 0.3017187, -0.5571567, 0.5484048
8: -0.2926740, 0.3067800, -0.2180972, 0.2138651, -0.5065392, 0.5248772
9: -0.1515706, 0.2054605, -0.1093603, 0.1627430, -0.3143137, 0.3148208

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 226

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8451745, upper bound: 1.8610458
time: 2.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8449334, upper bound: 1.8524917
time: 2.45 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1170018, 0.7372895, -0.1173968, 0.7201879, -0.8371897, 0.8546863
1: -0.2144630, 0.2429316, -0.2073334, 0.2420751, -0.4565381, 0.4502650
2: -0.3028396, 0.2845936, -0.2951292, 0.2847507, -0.5875903, 0.5797228
3: -0.2143469, 0.1250282, -0.2101713, 0.1226983, -0.3370451, 0.3351995
4: -0.1992715, 0.2909879, -0.1931328, 0.2896508, -0.4889223, 0.4841207
5: -0.3576107, 0.3723356, -0.3484295, 0.3695539, -0.7271646, 0.7207651
6: 0.0811579, 1.2629640, 0.0972198, 1.2597966, -1.1786387, 1.1657442
7: -0.2151300, 0.3510105, -0.2135950, 0.3440971, -0.5592270, 0.5646055
8: -0.2623310, 0.2679568, -0.2589912, 0.2670730, -0.5294040, 0.5269480
9: -0.1224668, 0.1864176, -0.1249440, 0.1866454, -0.3091122, 0.3113616

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8624133, upper bound: 1.8663752
time: 2.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8638346, upper bound: 1.8709439
time: 2.93 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1451652, 0.8233789, -0.1178418, 0.7404668, -0.8856320, 0.9412207
1: -0.2513816, 0.2814308, -0.2151092, 0.2514029, -0.5027844, 0.4965400
2: -0.3421652, 0.3312034, -0.3045909, 0.2946960, -0.6368612, 0.6357943
3: -0.2406964, 0.1658148, -0.2162493, 0.1297878, -0.3704842, 0.3820641
4: -0.2406279, 0.3220767, -0.2020621, 0.2968949, -0.5375229, 0.5241388
5: -0.3927745, 0.4119604, -0.3568623, 0.3772159, -0.7699904, 0.7688228
6: -0.0124801, 1.2759198, 0.0757660, 1.2618699, -1.2743500, 1.2001538
7: -0.2647755, 0.3858881, -0.2240977, 0.3520610, -0.6168365, 0.6099858
8: -0.2993227, 0.3155425, -0.2674959, 0.2770049, -0.5763276, 0.5830383
9: -0.1580051, 0.2104906, -0.1326584, 0.1911460, -0.3491511, 0.3431489

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8650061, upper bound: 1.8672226
time: 3.03 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8664904, upper bound: 1.8716409
time: 2.93 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1096684, 0.7003101, -0.1149822, 0.5852318, -0.6949002, 0.8152922
1: -0.1996840, 0.2301702, -0.1447565, 0.2039633, -0.4036473, 0.3749267
2: -0.2869682, 0.2579291, -0.2323117, 0.2192021, -0.5061703, 0.4902407
3: -0.2014219, 0.1138442, -0.1595826, 0.0951759, -0.2965978, 0.2734269
4: -0.1817832, 0.2761771, -0.1288218, 0.2492627, -0.4310459, 0.4049989
5: -0.3452377, 0.3476557, -0.2953548, 0.3054875, -0.6507252, 0.6430105
6: 0.1212656, 1.2467076, 0.2452843, 1.2494141, -1.1281484, 1.0014232
7: -0.1941680, 0.3363514, -0.1543139, 0.2808591, -0.4750271, 0.4906653
8: -0.2407732, 0.2415795, -0.2032897, 0.2057591, -0.4465322, 0.4448692
9: -0.1069796, 0.1712843, -0.1091306, 0.1561511, -0.2631307, 0.2804149

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8631241, upper bound: 1.8580985
time: 2.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8653176, upper bound: 1.8629667
time: 2.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1408186, 0.8286844, -0.1155242, 0.5908908, -0.7317094, 0.9442086
1: -0.2501911, 0.2744850, -0.1488633, 0.2066766, -0.4568677, 0.4233482
2: -0.3422541, 0.3238034, -0.2361032, 0.2198565, -0.5621107, 0.5599066
3: -0.2403390, 0.1572574, -0.1631698, 0.0963603, -0.3366994, 0.3204272
4: -0.2395738, 0.3155313, -0.1322325, 0.2521441, -0.4917179, 0.4477637
5: -0.3952719, 0.3997724, -0.2977467, 0.3105634, -0.7058353, 0.6975192
6: -0.0116338, 1.2738364, 0.2368209, 1.2505226, -1.2621565, 1.0370156
7: -0.2571230, 0.3851362, -0.1579090, 0.2852014, -0.5423244, 0.5430452
8: -0.2953991, 0.3065701, -0.2069836, 0.2064927, -0.5018918, 0.5135537
9: -0.1520708, 0.2030553, -0.1096638, 0.1581690, -0.3102398, 0.3127191

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8625343, upper bound: 1.8581000
time: 2.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8647309, upper bound: 1.8629681
time: 2.90 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1102243, 0.7266704, -0.1169259, 0.6386207, -0.7488450, 0.8435963
1: -0.2100578, 0.2359634, -0.1773131, 0.2231705, -0.4332283, 0.4132765
2: -0.2980923, 0.2702900, -0.2623827, 0.2459856, -0.5440779, 0.5326726
3: -0.2094967, 0.1196677, -0.1858702, 0.1051650, -0.3146617, 0.3055379
4: -0.1932022, 0.2826959, -0.1586159, 0.2694985, -0.4627007, 0.4413118
5: -0.3554358, 0.3574388, -0.3183998, 0.3401863, -0.6956222, 0.6758387
6: 0.0937470, 1.2507917, 0.1762908, 1.2551742, -1.1614271, 1.0745008
7: -0.2048527, 0.3464643, -0.1807986, 0.3138580, -0.5187107, 0.5272628
8: -0.2504264, 0.2535591, -0.2299402, 0.2298964, -0.4803228, 0.4834992
9: -0.1137831, 0.1769337, -0.1115415, 0.1696495, -0.2834325, 0.2884752

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8641006, upper bound: 1.8595905
time: 2.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8656430, upper bound: 1.8656245
time: 2.32 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1524276, 0.8550661, -0.1174702, 0.6497898, -0.8022174, 0.9725364
1: -0.2630671, 0.2847771, -0.1815518, 0.2260273, -0.4890944, 0.4663289
2: -0.3546189, 0.3367961, -0.2669355, 0.2523342, -0.6069531, 0.6037316
3: -0.2486141, 0.1714181, -0.1894313, 0.1078144, -0.3564285, 0.3608494
4: -0.2525975, 0.3246526, -0.1635554, 0.2727337, -0.5253313, 0.4882080
5: -0.4062799, 0.4137554, -0.3222253, 0.3451007, -0.7513806, 0.7359807
6: -0.0416307, 1.2850307, 0.1649903, 1.2567086, -1.2983394, 1.1200404
7: -0.2721728, 0.3970375, -0.1859658, 0.3180520, -0.5902249, 0.5830033
8: -0.3093170, 0.3204568, -0.2345919, 0.2359537, -0.5452707, 0.5550486
9: -0.1615783, 0.2106217, -0.1120750, 0.1726674, -0.3342456, 0.3226967

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8633944, upper bound: 1.8595892
time: 2.17 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8650101, upper bound: 1.8656251
time: 2.89 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1064422, 0.5658991, -0.1117931, 0.5185677, -0.6250099, 0.6776922
1: -0.1269894, 0.1873976, -0.1145721, 0.1779709, -0.3049603, 0.3019697
2: -0.2181839, 0.2075515, -0.2016091, 0.2163817, -0.4345656, 0.4091606
3: -0.1416411, 0.0883998, -0.1313636, 0.0812008, -0.2228419, 0.2197634
4: -0.1173297, 0.2308576, -0.1029899, 0.2223739, -0.3397036, 0.3338475
5: -0.2874044, 0.2719919, -0.2644463, 0.2628364, -0.5502408, 0.5364382
6: 0.2881253, 1.2298884, 0.3351195, 1.2436014, -0.9554761, 0.8947690
7: -0.1329810, 0.2642538, -0.1239611, 0.2411725, -0.3741535, 0.3882149
8: -0.1798689, 0.1936146, -0.1687407, 0.2024594, -0.3823283, 0.3623553
9: -0.1020379, 0.1413931, -0.1055103, 0.1394712, -0.2415091, 0.2469034

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8600839, upper bound: 1.8617538
time: 2.51 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8603928, upper bound: 1.8644414
time: 2.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1104228, 0.6998102, -0.1141077, 0.5845315, -0.6949543, 0.8139179
1: -0.1998707, 0.2304267, -0.1451153, 0.2044169, -0.4042876, 0.3755420
2: -0.2873771, 0.2575094, -0.2326429, 0.2179733, -0.5053504, 0.4901523
3: -0.2015198, 0.1139428, -0.1601557, 0.0948922, -0.2964119, 0.2740985
4: -0.1819753, 0.2763634, -0.1289419, 0.2495477, -0.4315230, 0.4053053
5: -0.3454007, 0.3484255, -0.2948478, 0.3055846, -0.6509852, 0.6432732
6: 0.1214511, 1.2485926, 0.2453108, 1.2471087, -1.1256577, 1.0032818
7: -0.1938484, 0.3365898, -0.1548691, 0.2817502, -0.4755986, 0.4914590
8: -0.2412092, 0.2416141, -0.2032486, 0.2044339, -0.4456432, 0.4448628
9: -0.1075058, 0.1714400, -0.1085195, 0.1562273, -0.2637331, 0.2799596

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8658496, upper bound: 1.8634092
time: 3.00 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8663449, upper bound: 1.8664510
time: 2.88 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1070644, 0.5706524, -0.1145134, 0.5759236, -0.6829880, 0.6851657
1: -0.1293181, 0.1903141, -0.1425761, 0.2038600, -0.3331781, 0.3328902
2: -0.2205355, 0.2083088, -0.2296021, 0.2189067, -0.4394423, 0.4379109
3: -0.1446284, 0.0896742, -0.1589535, 0.0936408, -0.2382692, 0.2486277
4: -0.1190050, 0.2338803, -0.1257550, 0.2491314, -0.3681363, 0.3596353
5: -0.2897004, 0.2766280, -0.2902344, 0.3051179, -0.5948184, 0.5668623
6: 0.2796341, 1.2311090, 0.2515205, 1.2482588, -0.9686247, 0.9795885
7: -0.1362216, 0.2677407, -0.1543775, 0.2791784, -0.4153999, 0.4221182
8: -0.1834879, 0.1943876, -0.2020822, 0.2053824, -0.3888703, 0.3964698
9: -0.1026449, 0.1434914, -0.1087099, 0.1566578, -0.2593027, 0.2522013

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8600952, upper bound: 1.8613777
time: 2.31 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8603942, upper bound: 1.8640084
time: 2.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1110798, 0.7125452, -0.1168905, 0.6679277, -0.7790076, 0.8294356
1: -0.2048300, 0.2337144, -0.1883591, 0.2301928, -0.4350228, 0.4220735
2: -0.2927069, 0.2648318, -0.2742600, 0.2619655, -0.5546725, 0.5390917
3: -0.2056181, 0.1170169, -0.1948957, 0.1116354, -0.3172535, 0.3119126
4: -0.1876946, 0.2800824, -0.1715636, 0.2772533, -0.4649478, 0.4516460
5: -0.3499758, 0.3541195, -0.3282813, 0.3514369, -0.7014127, 0.6824009
6: 0.1081367, 1.2507833, 0.1468951, 1.2558104, -1.1476736, 1.1038883
7: -0.1998177, 0.3414921, -0.1936916, 0.3250366, -0.5248542, 0.5351837
8: -0.2465983, 0.2485638, -0.2409671, 0.2448532, -0.4914516, 0.4895309
9: -0.1106449, 0.1749185, -0.1118284, 0.1765339, -0.2871788, 0.2867469

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8658769, upper bound: 1.8632797
time: 3.12 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8663487, upper bound: 1.8659546
time: 2.32 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1135554, 0.6360121, -0.1191000, 0.6106799, -0.7242354, 0.7551121
1: -0.1705644, 0.2162834, -0.1641832, 0.2169048, -0.3874692, 0.3804665
2: -0.2579608, 0.2302564, -0.2495917, 0.2325727, -0.4905335, 0.4798481
3: -0.1786533, 0.1033953, -0.1762586, 0.1008459, -0.2794992, 0.2796539
4: -0.1532454, 0.2616249, -0.1446501, 0.2631795, -0.4164248, 0.4062751
5: -0.3187860, 0.3260971, -0.3061878, 0.3302021, -0.6489881, 0.6322849
6: 0.1886147, 1.2491472, 0.2068576, 1.2581103, -1.0694957, 1.0422896
7: -0.1706919, 0.3074552, -0.1711579, 0.3006751, -0.4713670, 0.4786132
8: -0.2208689, 0.2161307, -0.2211658, 0.2172328, -0.4381016, 0.4372966
9: -0.1087967, 0.1621029, -0.1125679, 0.1663459, -0.2751426, 0.2746708

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 226

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8448008, upper bound: 1.8549183
time: 2.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8445931, upper bound: 1.8430301
time: 2.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1167511, 0.7461733, -0.1197289, 0.6179211, -0.7346722, 0.8659022
1: -0.2178581, 0.2451593, -0.1685856, 0.2196731, -0.4375312, 0.4137449
2: -0.3063895, 0.2898782, -0.2534205, 0.2387867, -0.5451761, 0.5432986
3: -0.2170255, 0.1270563, -0.1799674, 0.1020489, -0.3190745, 0.3070238
4: -0.2032790, 0.2933089, -0.1493215, 0.2661108, -0.4693897, 0.4426304
5: -0.3604260, 0.3755799, -0.3092766, 0.3353700, -0.6957960, 0.6848564
6: 0.0717118, 1.2626547, 0.1973586, 1.2596407, -1.1879289, 1.0652961
7: -0.2194524, 0.3543439, -0.1748883, 0.3050668, -0.5245191, 0.5292323
8: -0.2656045, 0.2729171, -0.2249249, 0.2231349, -0.4887394, 0.4978420
9: -0.1260183, 0.1885911, -0.1131131, 0.1684070, -0.2944253, 0.3017042

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 226

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8448765, upper bound: 1.8569802
time: 2.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8446610, upper bound: 1.8431609
time: 9.32 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1140044, 0.6500255, -0.1211205, 0.6922856, -0.8062900, 0.7711459
1: -0.1794131, 0.2211369, -0.1974985, 0.2375266, -0.4169397, 0.4186354
2: -0.2660866, 0.2385291, -0.2840440, 0.2772582, -0.5433449, 0.5225731
3: -0.1855554, 0.1059128, -0.2031521, 0.1181702, -0.3037256, 0.3090649
4: -0.1608803, 0.2666913, -0.1823604, 0.2857637, -0.4466439, 0.4490517
5: -0.3251774, 0.3348252, -0.3370392, 0.3651647, -0.6903421, 0.6718645
6: 0.1706344, 1.2513858, 0.1224827, 1.2653725, -1.0947381, 1.1289032
7: -0.1771578, 0.3163415, -0.2060819, 0.3339008, -0.5110586, 0.5224234
8: -0.2275781, 0.2240476, -0.2532138, 0.2601250, -0.4877030, 0.4772614
9: -0.1094006, 0.1652823, -0.1205035, 0.1851404, -0.2945411, 0.2857858

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8652762, upper bound: 1.8605947
time: 2.31 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8669696, upper bound: 1.8664624
time: 3.09 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1233373, 0.7708169, -0.1217968, 0.7037569, -0.8270941, 0.8926138
1: -0.2273784, 0.2548822, -0.2018125, 0.2422054, -0.4695838, 0.4566947
2: -0.3176291, 0.3011509, -0.2890562, 0.2836739, -0.6013029, 0.5902070
3: -0.2243537, 0.1347178, -0.2067526, 0.1217756, -0.3461294, 0.3414704
4: -0.2140067, 0.3011998, -0.1874895, 0.2897762, -0.5037829, 0.4886893
5: -0.3707425, 0.3844975, -0.3413387, 0.3701698, -0.7409123, 0.7258362
6: 0.0460394, 1.2663264, 0.1106862, 1.2669914, -1.2209520, 1.1556401
7: -0.2310336, 0.3639706, -0.2120599, 0.3383241, -0.5693576, 0.5760305
8: -0.2752913, 0.2843079, -0.2582840, 0.2665117, -0.5418030, 0.5425919
9: -0.1342405, 0.1936986, -0.1252076, 0.1882174, -0.3224579, 0.3189062

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8648316, upper bound: 1.8605960
time: 3.20 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8664664, upper bound: 1.8664664
time: 2.52 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 7.01 seconds
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8588657, upper bound: 1.8647502
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8590809, upper bound: 1.8666937
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8647601, upper bound: 1.8682885
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8650325, upper bound: 1.8702528
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8612718, upper bound: 1.8639292
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8626877, upper bound: 1.8688729
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8635809, upper bound: 1.8648069
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8650357, upper bound: 1.8696955
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8601372, upper bound: 1.8654675
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8604160, upper bound: 1.8674452
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8659501, upper bound: 1.8689603
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8663704, upper bound: 1.8710082
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8601392, upper bound: 1.8649158
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8604171, upper bound: 1.8668605
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8659638, upper bound: 1.8686813
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8663737, upper bound: 1.8704570
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8433373, upper bound: 1.8599637
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8430796, upper bound: 1.8510437
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8451745, upper bound: 1.8610458
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8449334, upper bound: 1.8524917
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8624133, upper bound: 1.8663752
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8638346, upper bound: 1.8709439
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8650061, upper bound: 1.8672226
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8664904, upper bound: 1.8716409
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8631241, upper bound: 1.8580985
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8653176, upper bound: 1.8629667
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8625343, upper bound: 1.8581000
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8647309, upper bound: 1.8629681
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8641006, upper bound: 1.8595905
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8656430, upper bound: 1.8656245
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8633944, upper bound: 1.8595892
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8650101, upper bound: 1.8656251
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8600839, upper bound: 1.8617538
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8603928, upper bound: 1.8644414
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8658496, upper bound: 1.8634092
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8663449, upper bound: 1.8664510
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8600952, upper bound: 1.8613777
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8603942, upper bound: 1.8640084
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8658769, upper bound: 1.8632797
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8663487, upper bound: 1.8659546
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8448008, upper bound: 1.8549183
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8445931, upper bound: 1.8430301
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8448765, upper bound: 1.8569802
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8446610, upper bound: 1.8431609
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8652762, upper bound: 1.8605947
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8669696, upper bound: 1.8664624
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8648316, upper bound: 1.8605960
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.01
Output dim: 6, lower bound: -1.8664664, upper bound: 1.8664664

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1060537, 0.5519612, -0.1046499, 0.4419908, -0.5480445, 0.6566111
1: -0.1229085, 0.1841320, -0.0946776, 0.1203581, -0.2432666, 0.2788095
2: -0.2127815, 0.2074855, -0.1566257, 0.2094137, -0.4221952, 0.3641112
3: -0.1374131, 0.0856063, -0.0971745, 0.0668494, -0.2042626, 0.1827807
4: -0.1128149, 0.2276177, -0.0710256, 0.1735113, -0.2863262, 0.2986432
5: -0.2802165, 0.2674890, -0.2297340, 0.2034296, -0.4836461, 0.4972230
6: 0.3036733, 1.2292026, 0.4865753, 1.2282579, -0.9245846, 0.7426273
7: -0.1298980, 0.2567190, -0.1175252, 0.1771582, -0.3070562, 0.3742442
8: -0.1751783, 0.1934762, -0.1402249, 0.1929955, -0.3681739, 0.3337011
9: -0.1015531, 0.1398259, -0.0977802, 0.1048198, -0.2063729, 0.2376060

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8552050, upper bound: 1.8526632
time: 2.24 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8468818, upper bound: 1.8524960
time: 2.89 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1065303, 0.5639716, -0.1058442, 0.4663416, -0.5728719, 0.6698158
1: -0.1276264, 0.1893187, -0.0995475, 0.1390907, -0.2667171, 0.2888662
2: -0.2180394, 0.2078179, -0.1687794, 0.2101634, -0.4282027, 0.3765973
3: -0.1431047, 0.0882993, -0.1064470, 0.0710742, -0.2141790, 0.1947463
4: -0.1168421, 0.2329068, -0.0798873, 0.1865346, -0.3033767, 0.3127941
5: -0.2861605, 0.2750671, -0.2396701, 0.2201321, -0.5062926, 0.5147372
6: 0.2859861, 1.2299016, 0.4384674, 1.2306507, -0.9446645, 0.7914342
7: -0.1354882, 0.2643425, -0.1184294, 0.1938073, -0.3292955, 0.3827719
8: -0.1816079, 0.1938856, -0.1478890, 0.1943878, -0.3759957, 0.3417746
9: -0.1021766, 0.1431059, -0.0992473, 0.1156328, -0.2178094, 0.2423532

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8557204, upper bound: 1.8543581
time: 2.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8471026, upper bound: 1.8541784
time: 2.94 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1100327, 0.6820347, -0.1062030, 0.4900818, -0.6001145, 0.7882376
1: -0.1922087, 0.2274082, -0.1040293, 0.1471392, -0.3393479, 0.3314375
2: -0.2789901, 0.2534598, -0.1781972, 0.2099704, -0.4889606, 0.4316570
3: -0.1964813, 0.1104408, -0.1102845, 0.0735435, -0.2700248, 0.2207252
4: -0.1745074, 0.2731602, -0.0874922, 0.1932519, -0.3677593, 0.3606524
5: -0.3371634, 0.3433287, -0.2485771, 0.2301470, -0.5673103, 0.5919058
6: 0.1398654, 1.2443538, 0.4094012, 1.2311625, -1.0912971, 0.8349525
7: -0.1892510, 0.3291903, -0.1184960, 0.2067546, -0.3960056, 0.4476863
8: -0.2358935, 0.2366613, -0.1529564, 0.1943409, -0.4302344, 0.3896177
9: -0.1070084, 0.1694949, -0.0998085, 0.1188837, -0.2258921, 0.2693034

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 226

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8400081, upper bound: 1.8339672
time: 2.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8234147, upper bound: 1.8302463
time: 2.54 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1105256, 0.7051613, -0.1075786, 0.5156544, -0.6261801, 0.8127400
1: -0.2013441, 0.2325656, -0.1106718, 0.1652147, -0.3665589, 0.3432375
2: -0.2887912, 0.2643642, -0.1938173, 0.2107592, -0.4995504, 0.4581816
3: -0.2035419, 0.1155550, -0.1217929, 0.0776989, -0.2812409, 0.2373479
4: -0.1845923, 0.2789394, -0.0984837, 0.2095659, -0.3941582, 0.3774231
5: -0.3461233, 0.3519557, -0.2603803, 0.2490807, -0.5952040, 0.6123360
6: 0.1159745, 1.2474496, 0.3605212, 1.2338011, -1.1178266, 0.8869284
7: -0.1986040, 0.3381355, -0.1195922, 0.2292415, -0.4278455, 0.4577278
8: -0.2443981, 0.2472940, -0.1603715, 0.1960386, -0.4404367, 0.4076655
9: -0.1112500, 0.1744463, -0.1017639, 0.1292925, -0.2405424, 0.2762103

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 226

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412133, upper bound: 1.8355521
time: 2.24 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8237401, upper bound: 1.8318701
time: 3.02 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1080543, 0.6057809, -0.1061459, 0.4490721, -0.5571264, 0.7119268
1: -0.1504395, 0.2039077, -0.0960981, 0.1320854, -0.2825249, 0.3000058
2: -0.2388983, 0.2100396, -0.1642671, 0.2112886, -0.4501869, 0.3743067
3: -0.1630188, 0.0960358, -0.1038506, 0.0692139, -0.2322327, 0.1998864
4: -0.1357594, 0.2481822, -0.0741320, 0.1811303, -0.3168898, 0.3223141
5: -0.3047926, 0.3015964, -0.2326550, 0.2123031, -0.5170957, 0.5342515
6: 0.2320872, 1.2338890, 0.4610759, 1.2317731, -0.9996859, 0.7728131
7: -0.1548632, 0.2870903, -0.1189099, 0.1842701, -0.3391333, 0.4060001
8: -0.2020896, 0.1953330, -0.1443181, 0.1954093, -0.3974989, 0.3396511
9: -0.1041610, 0.1523487, -0.0991836, 0.1127509, -0.2169119, 0.2515324

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8571318, upper bound: 1.8523564
time: 2.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8497207, upper bound: 1.8521708
time: 2.24 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1106291, 0.6911169, -0.1087764, 0.5304372, -0.6410663, 0.7998933
1: -0.1956763, 0.2293900, -0.1163336, 0.1769679, -0.3726442, 0.3457236
2: -0.2827411, 0.2572946, -0.2036949, 0.2118906, -0.4946317, 0.4609895
3: -0.1991732, 0.1124854, -0.1299397, 0.0818702, -0.2810434, 0.2424252
4: -0.1782065, 0.2754198, -0.1054125, 0.2210147, -0.3992213, 0.3808323
5: -0.3407462, 0.3469447, -0.2689313, 0.2613703, -0.6021165, 0.6158760
6: 0.1308115, 1.2464521, 0.3292936, 1.2361803, -1.1053689, 0.9171585
7: -0.1927172, 0.3325056, -0.1235768, 0.2439353, -0.4366525, 0.4560824
8: -0.2393633, 0.2405186, -0.1683001, 0.1976829, -0.4370462, 0.4088186
9: -0.1075533, 0.1714612, -0.1032391, 0.1369308, -0.2444841, 0.2747003

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 226

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8416658, upper bound: 1.8345972
time: 2.23 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8218384, upper bound: 1.8308148
time: 2.28 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1101773, 0.6589786, -0.1063940, 0.4564770, -0.5666543, 0.7653726
1: -0.1833198, 0.2229656, -0.0976571, 0.1370143, -0.3203341, 0.3206227
2: -0.2694535, 0.2445847, -0.1674078, 0.2113730, -0.4808265, 0.4119925
3: -0.1898297, 0.1057785, -0.1063021, 0.0704002, -0.2602299, 0.2120806
4: -0.1649647, 0.2682376, -0.0769533, 0.1846935, -0.3496582, 0.3451909
5: -0.3275860, 0.3362655, -0.2356611, 0.2175956, -0.5451815, 0.5719266
6: 0.1626631, 1.2423774, 0.4474878, 1.2322201, -1.0695570, 0.7948896
7: -0.1812363, 0.3203362, -0.1190915, 0.1891915, -0.3704278, 0.4394277
8: -0.2285421, 0.2279175, -0.1464987, 0.1956674, -0.4242095, 0.3744162
9: -0.1068194, 0.1657299, -0.0995258, 0.1154452, -0.2222647, 0.2652557

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 226

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8402031, upper bound: 1.8311570
time: 2.28 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8229640, upper bound: 1.8278082
time: 2.37 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1230886, 0.7778752, -0.1091080, 0.5378158, -0.6609043, 0.8869832
1: -0.2286421, 0.2547349, -0.1192876, 0.1811015, -0.4097435, 0.3740225
2: -0.3194496, 0.3006719, -0.2077234, 0.2121175, -0.5315671, 0.5083953
3: -0.2256182, 0.1344868, -0.1332936, 0.0840345, -0.3096527, 0.2677804
4: -0.2159204, 0.2999518, -0.1084444, 0.2251150, -0.4410354, 0.4083962
5: -0.3737648, 0.3801265, -0.2734826, 0.2658385, -0.6396033, 0.6536092
6: 0.0422930, 1.2591381, 0.3166625, 1.2366595, -1.1943666, 0.9424756
7: -0.2313921, 0.3653497, -0.1272174, 0.2499348, -0.4813269, 0.4925670
8: -0.2727584, 0.2825738, -0.1727006, 0.1979633, -0.4707217, 0.4552744
9: -0.1351029, 0.1912265, -0.1036793, 0.1395362, -0.2746390, 0.2949058

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 226

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8438562, upper bound: 1.8355672
time: 2.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8238060, upper bound: 1.8318701
time: 2.20 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1055084, 0.5513017, -0.1047737, 0.4430409, -0.5485493, 0.6560754
1: -0.1201941, 0.1762142, -0.0948681, 0.1203433, -0.2405373, 0.2710823
2: -0.2086100, 0.2068634, -0.1567136, 0.2095596, -0.4181696, 0.3635770
3: -0.1334537, 0.0835932, -0.0973200, 0.0669228, -0.2003765, 0.1809132
4: -0.1103841, 0.2198541, -0.0712705, 0.1736183, -0.2840024, 0.2911245
5: -0.2774998, 0.2607881, -0.2299859, 0.2037719, -0.4812717, 0.4907740
6: 0.3178962, 1.2285209, 0.4860409, 1.2285305, -0.9106343, 0.7424799
7: -0.1235076, 0.2500226, -0.1176052, 0.1776011, -0.3011086, 0.3676278
8: -0.1699547, 0.1926189, -0.1404534, 0.1931160, -0.3630708, 0.3330722
9: -0.1007258, 0.1339480, -0.0978785, 0.1047248, -0.2054507, 0.2318265

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 226

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8393006, upper bound: 1.8335274
time: 2.15 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8241654, upper bound: 1.8313922
time: 2.49 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1058205, 0.5593972, -0.1059597, 0.4673394, -0.5731599, 0.6653568
1: -0.1232248, 0.1806088, -0.0996974, 0.1390109, -0.2622357, 0.2803062
2: -0.2130287, 0.2069698, -0.1689789, 0.2103118, -0.4233406, 0.3759487
3: -0.1370420, 0.0857940, -0.1065502, 0.0711314, -0.2081735, 0.1923442
4: -0.1136769, 0.2241612, -0.0800803, 0.1865947, -0.3002715, 0.3042414
5: -0.2824558, 0.2654895, -0.2399152, 0.2202760, -0.5027317, 0.5054047
6: 0.3045716, 1.2290530, 0.4381229, 1.2309146, -0.9263430, 0.7909301
7: -0.1272376, 0.2563905, -0.1185017, 0.1941970, -0.3214346, 0.3748922
8: -0.1736672, 0.1929460, -0.1480880, 0.1945004, -0.3681676, 0.3410340
9: -0.1012177, 0.1367121, -0.0993317, 0.1155046, -0.2167223, 0.2360439

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 226

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8398371, upper bound: 1.8343695
time: 2.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8242358, upper bound: 1.8321329
time: 2.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1093891, 0.6629473, -0.1063291, 0.4916500, -0.6010391, 0.7692764
1: -0.1820185, 0.2196375, -0.1042958, 0.1473300, -0.3293484, 0.3239333
2: -0.2695735, 0.2360970, -0.1786779, 0.2101150, -0.4796886, 0.4147749
3: -0.1870087, 0.1058474, -0.1105167, 0.0736712, -0.2606799, 0.2163641
4: -0.1647176, 0.2645429, -0.0878930, 0.1935125, -0.3582301, 0.3524359
5: -0.3313477, 0.3297801, -0.2490421, 0.2305922, -0.5619399, 0.5788222
6: 0.1636258, 1.2435660, 0.4082461, 1.2314425, -1.0678166, 0.8353200
7: -0.1759746, 0.3188954, -0.1185704, 0.2075000, -0.3834746, 0.4374658
8: -0.2248942, 0.2216745, -0.1532860, 0.1944543, -0.4193485, 0.3749605
9: -0.1061268, 0.1618041, -0.0999101, 0.1188906, -0.2250174, 0.2617142

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 226

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8444067, upper bound: 1.8371801
time: 1.98 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8301947, upper bound: 1.8352623
time: 2.59 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 5.86 seconds
IS_A1_B1_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8552050, upper bound: 1.8526632
IS_A1_B1_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8468818, upper bound: 1.8524960
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8557204, upper bound: 1.8543581
IS_A1_B1_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8471026, upper bound: 1.8541784
IS_A1_B1_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8400081, upper bound: 1.8339672
IS_A1_B1_A1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8234147, upper bound: 1.8302463
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8412133, upper bound: 1.8355521
IS_A1_B1_A1_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8237401, upper bound: 1.8318701
IS_A1_B1_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8571318, upper bound: 1.8523564
IS_A1_B1_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8497207, upper bound: 1.8521708
IS_A1_B1_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8416658, upper bound: 1.8345972
IS_A1_B1_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8218384, upper bound: 1.8308148
IS_A1_B1_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8402031, upper bound: 1.8311570
IS_A1_B1_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8229640, upper bound: 1.8278082
IS_A1_B1_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8438562, upper bound: 1.8355672
IS_A1_B1_A1_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8238060, upper bound: 1.8318701
IS_A1_B1_A2_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8393006, upper bound: 1.8335274
IS_A1_B1_A2_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8241654, upper bound: 1.8313922
IS_A1_B1_A2_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8398371, upper bound: 1.8343695
IS_A1_B1_A2_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8242358, upper bound: 1.8321329
IS_A1_B1_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8444067, upper bound: 1.8371801
IS_A1_B1_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 5.86
Output dim: 6, lower bound: -1.8301947, upper bound: 1.8352623
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8663704, upper bound: 1.8710082
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8601392, upper bound: 1.8649158
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8604171, upper bound: 1.8668605
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8659638, upper bound: 1.8686813
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8663737, upper bound: 1.8704570
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8433373, upper bound: 1.8599637
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8430796, upper bound: 1.8510437
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8451745, upper bound: 1.8610458
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8449334, upper bound: 1.8524917
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8624133, upper bound: 1.8663752
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8638346, upper bound: 1.8709439
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8650061, upper bound: 1.8672226
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8664904, upper bound: 1.8716409
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8631241, upper bound: 1.8580985
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8653176, upper bound: 1.8629667
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8625343, upper bound: 1.8581000
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8647309, upper bound: 1.8629681
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8641006, upper bound: 1.8595905
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8656430, upper bound: 1.8656245
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8633944, upper bound: 1.8595892
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8650101, upper bound: 1.8656251
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8600839, upper bound: 1.8617538
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8603928, upper bound: 1.8644414
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8658496, upper bound: 1.8634092
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8663449, upper bound: 1.8664510
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8600952, upper bound: 1.8613777
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8603942, upper bound: 1.8640084
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8658769, upper bound: 1.8632797
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8663487, upper bound: 1.8659546
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8448008, upper bound: 1.8549183
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8445931, upper bound: 1.8430301
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8448765, upper bound: 1.8569802
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8446610, upper bound: 1.8431609
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8652762, upper bound: 1.8605947
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8669696, upper bound: 1.8664624
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8648316, upper bound: 1.8605960
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.86
Output dim: 6, lower bound: -1.8664664, upper bound: 1.8664664
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.094357967376709
rel_dist={6: [-1.90967747725962, 1.9096774281054198]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 1828.17 seconds
