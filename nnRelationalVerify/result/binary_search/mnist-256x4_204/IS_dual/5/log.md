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
execution time: IAR + LP analysis = 1.12 + 2.89 = 4.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -1.9162456, upper bound: 1.9162456


# Binary Search by BASE starts (time budget: 2695.99 seconds, max iter: 100)

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
Binary search time: 15.44 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2680.55 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 189

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9092793, upper bound: 1.8660586
time: 1.67 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877
time: 2.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 4.39 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 4.39
Output dim: 6, lower bound: -1.9092793, upper bound: 1.8660586
IS_A2, status: Status.UNKNOWN, split count: 1, time: 4.39
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

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 189

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877
time: 2.03 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877
time: 2.31 seconds

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

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8484793, upper bound: 1.8313277
time: 1.78 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8312450, upper bound: 1.8312450
time: 1.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.57 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.57
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.57
Output dim: 6, lower bound: -1.8654877, upper bound: 1.8654877
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.57
Output dim: 6, lower bound: -1.8484793, upper bound: 1.8313277
IS_A2_A2, status: Status.VERIFIED, split count: 2, time: 4.57
Output dim: 6, lower bound: -1.8312450, upper bound: 1.8312450

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

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8900152, upper bound: 1.8319765
time: 1.95 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8843870, upper bound: 1.8319511
time: 2.12 seconds

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

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8861910, upper bound: 1.8494081
time: 2.02 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8843870, upper bound: 1.8319511
time: 1.76 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -0.8361796, 1.6916373, -0.4592743, 1.3080159, -2.1441956, 2.1509116
1: -0.7908013, 0.7925454, -0.5311626, 0.5342227, -1.3250241, 1.3237081
2: -0.8674313, 1.0445075, -0.6175320, 0.6685661, -1.5359974, 1.6620395
3: -0.6669739, 0.8051766, -0.4319794, 0.4902656, -1.1572396, 1.2371560
4: -0.8164440, 1.1000400, -0.5498180, 0.6282432, -1.4446871, 1.6498580
5: -1.0171378, 1.1558790, -0.6556945, 0.7794052, -1.7965429, 1.8115735
6: -1.0718563, 1.6044312, -0.5814891, 1.4815513, -2.5534077, 2.1859202
7: -0.9843407, 0.9428054, -0.6174742, 0.6684646, -1.6528053, 1.5602796
8: -0.8967382, 1.1181587, -0.5971164, 0.6973907, -1.5941288, 1.7152750
9: -0.7806149, 0.9022418, -0.4699131, 0.5248307, -1.3054456, 1.3721548

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 189

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8484793, upper bound: 1.8313277
time: 1.95 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8484793, upper bound: 1.8313277
time: 1.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.92 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.92
Output dim: 6, lower bound: -1.8900152, upper bound: 1.8319765
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.92
Output dim: 6, lower bound: -1.8843870, upper bound: 1.8319511
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.92
Output dim: 6, lower bound: -1.8861910, upper bound: 1.8494081
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.92
Output dim: 6, lower bound: -1.8843870, upper bound: 1.8319511
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.92
Output dim: 6, lower bound: -1.8484793, upper bound: 1.8313277
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.92
Output dim: 6, lower bound: -1.8484793, upper bound: 1.8313277

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

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8991594, upper bound: 1.8900493
time: 2.06 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8991594, upper bound: 1.8905728
time: 2.46 seconds

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

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8904415, upper bound: 1.8899819
time: 2.01 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8905455, upper bound: 1.8905455
time: 2.22 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.4283954, 1.2709696, -0.8361796, 1.6916373, -2.1200328, 2.1071491
1: -0.5065504, 0.5120057, -0.7908013, 0.7925454, -1.2990959, 1.3028071
2: -0.5936686, 0.6343641, -0.8674313, 1.0445075, -1.6381761, 1.5017955
3: -0.4099944, 0.4626325, -0.6669739, 0.8051766, -1.2151711, 1.1296065
4: -0.5260350, 0.5839709, -0.8164440, 1.1000400, -1.6260750, 1.4004149
5: -0.6253715, 0.7425824, -1.0171378, 1.1558790, -1.7812505, 1.7597201
6: -0.5344308, 1.4685735, -1.0718563, 1.6044312, -2.1388619, 2.5404296
7: -0.5838713, 0.6439485, -0.9843407, 0.9428054, -1.5266767, 1.6282892
8: -0.5708863, 0.6576684, -0.8967382, 1.1181587, -1.6890450, 1.5544065
9: -0.4413340, 0.4900914, -0.7806149, 0.9022418, -1.3435758, 1.2707063

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8826493, upper bound: 1.8482865
time: 1.72 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8861910, upper bound: 1.8494081
time: 2.03 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.3830998, 1.1939206, -0.8446211, 1.6987628, -2.0818624, 2.0385418
1: -0.4624637, 0.4786091, -0.7967445, 0.7988135, -1.2612772, 1.2753536
2: -0.5521010, 0.5773075, -0.8728036, 1.0532151, -1.6053162, 1.4501110
3: -0.3769137, 0.4171165, -0.6775891, 0.8122519, -1.1891656, 1.0947056
4: -0.4822236, 0.5206205, -0.8219454, 1.1102685, -1.5924921, 1.3425660
5: -0.5755689, 0.6879572, -1.0248938, 1.1673429, -1.7429118, 1.7128509
6: -0.4452221, 1.4325594, -1.0824918, 1.6062536, -2.0514758, 2.5150511
7: -0.5307168, 0.6035373, -0.9924264, 0.9492077, -1.4799244, 1.5959637
8: -0.5228423, 0.5970218, -0.9059386, 1.1287196, -1.6515620, 1.5029603
9: -0.3955613, 0.4396504, -0.7863836, 0.9114796, -1.3070409, 1.2260339

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8774045, upper bound: 1.8248551
time: 1.67 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8843870, upper bound: 1.8319511
time: 1.95 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.8361796, 1.6916373, -0.4283954, 1.2709696, -2.1071491, 2.1200328
1: -0.7908013, 0.7925454, -0.5065504, 0.5120057, -1.3028071, 1.2990959
2: -0.8674313, 1.0445075, -0.5936686, 0.6343641, -1.5017955, 1.6381761
3: -0.6669739, 0.8051766, -0.4099944, 0.4626325, -1.1296065, 1.2151711
4: -0.8164440, 1.1000400, -0.5260350, 0.5839709, -1.4004149, 1.6260750
5: -1.0171378, 1.1558790, -0.6253715, 0.7425824, -1.7597201, 1.7812505
6: -1.0718563, 1.6044312, -0.5344308, 1.4685735, -2.5404296, 2.1388619
7: -0.9843407, 0.9428054, -0.5838713, 0.6439485, -1.6282892, 1.5266767
8: -0.8967382, 1.1181587, -0.5708863, 0.6576684, -1.5544065, 1.6890450
9: -0.7806149, 0.9022418, -0.4413340, 0.4900914, -1.2707063, 1.3435758

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8473163, upper bound: 1.8306415
time: 1.89 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8484793, upper bound: 1.8313277
time: 1.78 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.8361796, 1.6916373, -1.1578722, 2.0902753, -2.9264550, 2.8495095
1: -0.7908013, 0.7925454, -1.0186968, 0.9981120, -1.7889132, 1.8112422
2: -0.8674313, 1.0445075, -1.0924278, 1.3473158, -2.2147472, 2.1369352
3: -0.6669739, 0.8051766, -0.8712537, 1.0596817, -1.7266556, 1.6764302
4: -0.8164440, 1.1000400, -1.0514562, 1.4630030, -2.2794471, 2.1514962
5: -1.0171378, 1.1558790, -1.3404843, 1.4613965, -2.4785342, 2.4963632
6: -1.0718563, 1.6044312, -1.5363673, 1.8078189, -2.8796751, 3.1407986
7: -0.9843407, 0.9428054, -1.2797452, 1.1851227, -2.1694634, 2.2225506
8: -0.8967382, 1.1181587, -1.1523812, 1.4595408, -2.3562789, 2.2705398
9: -0.7806149, 0.9022418, -1.0169351, 1.1958456, -1.9764605, 1.9191768

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7017395, upper bound: 1.5550660
time: 2.30 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.5996925, upper bound: 1.5507723
time: 1.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.14 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 6, lower bound: -1.8991594, upper bound: 1.8900493
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 6, lower bound: -1.8991594, upper bound: 1.8905728
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 6, lower bound: -1.8904415, upper bound: 1.8899819
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 6, lower bound: -1.8905455, upper bound: 1.8905455
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 6, lower bound: -1.8826493, upper bound: 1.8482865
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 6, lower bound: -1.8861910, upper bound: 1.8494081
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 6, lower bound: -1.8774045, upper bound: 1.8248551
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 6, lower bound: -1.8843870, upper bound: 1.8319511
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 6, lower bound: -1.8473163, upper bound: 1.8306415
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.14
Output dim: 6, lower bound: -1.8484793, upper bound: 1.8313277
IS_A2_A1_B2_A1, status: Status.VERIFIED, split count: 4, time: 5.14
Output dim: 6, lower bound: -1.7017395, upper bound: 1.5550660
IS_A2_A1_B2_A2, status: Status.VERIFIED, split count: 4, time: 5.14
Output dim: 6, lower bound: -1.5996925, upper bound: 1.5507723

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2154518, 0.9344941, -0.4096891, 1.2453507, -1.4608026, 1.3441832
1: -0.3147149, 0.3367910, -0.4897345, 0.4982932, -0.8130081, 0.8265255
2: -0.4023729, 0.4023019, -0.5781872, 0.6110476, -1.0134205, 0.9804891
3: -0.2816463, 0.2394949, -0.3958672, 0.4440157, -0.7256620, 0.6353621
4: -0.3105045, 0.3721581, -0.5098313, 0.5555456, -0.8660501, 0.8819894
5: -0.4456833, 0.4830772, -0.6069617, 0.7180941, -1.1637774, 1.0900389
6: -0.1440101, 1.3136120, -0.5021762, 1.4600376, -1.6040477, 1.8157883
7: -0.3404821, 0.4518234, -0.5615412, 0.6287183, -0.9692004, 1.0133646
8: -0.3618510, 0.3986605, -0.5532811, 0.6309687, -0.9928197, 0.9519416
9: -0.2250141, 0.2665265, -0.4216886, 0.4685784, -0.6935925, 0.6882150

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8991594, upper bound: 1.8900493
time: 2.42 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8991594, upper bound: 1.8900493
time: 2.16 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.2072586, 0.8959078, -0.4283954, 1.2709696, -1.4782282, 1.3243032
1: -0.3009212, 0.3340663, -0.5065504, 0.5120057, -0.8129269, 0.8406167
2: -0.3868112, 0.3966483, -0.5936686, 0.6343641, -1.0211754, 0.9903169
3: -0.2720728, 0.2352810, -0.4099944, 0.4626325, -0.7347053, 0.6452754
4: -0.2967181, 0.3695163, -0.5260350, 0.5839709, -0.8806890, 0.8955513
5: -0.4288297, 0.4805791, -0.6253715, 0.7425824, -1.1714121, 1.1059506
6: -0.1073784, 1.2879556, -0.5344308, 1.4685735, -1.5759518, 1.8223865
7: -0.3334463, 0.4387536, -0.5838713, 0.6439485, -0.9773948, 1.0226250
8: -0.3504367, 0.3934267, -0.5708863, 0.6576684, -1.0081050, 0.9643130
9: -0.2233384, 0.2670660, -0.4413340, 0.4900914, -0.7134299, 0.7084001

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8991593, upper bound: 1.8905728
time: 2.20 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8991593, upper bound: 1.8905728
time: 2.52 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2265769, 0.9415536, -0.3679794, 1.1703347, -1.3969116, 1.3095330
1: -0.3216501, 0.3471616, -0.4484968, 0.4651441, -0.7867942, 0.7956583
2: -0.4083511, 0.4149002, -0.5383945, 0.5603703, -0.9687214, 0.9532946
3: -0.2864379, 0.2512243, -0.3685306, 0.3987908, -0.6852287, 0.6197549
4: -0.3193513, 0.3821320, -0.4662998, 0.5012539, -0.8206053, 0.8484317
5: -0.4501562, 0.4991343, -0.5622077, 0.6684925, -1.1186486, 1.0613420
6: -0.1542069, 1.3141832, -0.4178757, 1.4242272, -1.5784342, 1.7320590
7: -0.3526507, 0.4599188, -0.5121202, 0.5898783, -0.9425290, 0.9720391
8: -0.3686532, 0.4127948, -0.5071126, 0.5779853, -0.9466385, 0.9199075
9: -0.2368645, 0.2806606, -0.3775311, 0.4217539, -0.6586185, 0.6581917

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8904415, upper bound: 1.8899819
time: 2.10 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8904415, upper bound: 1.8899819
time: 2.19 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.2095052, 0.8895733, -0.3830998, 1.1939206, -1.4034258, 1.2726730
1: -0.3008616, 0.3373545, -0.4624637, 0.4786091, -0.7794707, 0.7998182
2: -0.3856023, 0.4003984, -0.5521010, 0.5773075, -0.9629098, 0.9524994
3: -0.2718897, 0.2389219, -0.3769137, 0.4171165, -0.6890062, 0.6158355
4: -0.2969734, 0.3729972, -0.4822236, 0.5206205, -0.8175939, 0.8552208
5: -0.4263875, 0.4867312, -0.5755689, 0.6879572, -1.1143447, 1.0623001
6: -0.1041598, 1.2882056, -0.4452221, 1.4325594, -1.5367192, 1.7334278
7: -0.3368429, 0.4387551, -0.5307168, 0.6035373, -0.9403802, 0.9694718
8: -0.3541161, 0.3981440, -0.5228423, 0.5970218, -0.9511378, 0.9209863
9: -0.2273098, 0.2730464, -0.3955613, 0.4396504, -0.6669602, 0.6686077

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8905455, upper bound: 1.8905455
time: 2.39 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8905455, upper bound: 1.8905455
time: 2.37 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.4067150, 1.2407022, -0.8261355, 1.6957741, -2.1024890, 2.0668375
1: -0.4869262, 0.4960527, -0.7845350, 0.7813374, -1.2682636, 1.2805877
2: -0.5755167, 0.6072444, -0.8624935, 1.0301049, -1.6056216, 1.4697378
3: -0.3936818, 0.4409779, -0.6554159, 0.7940063, -1.1876881, 1.0963938
4: -0.5070575, 0.5511866, -0.8107981, 1.0799718, -1.5870293, 1.3619847
5: -0.6038042, 0.7142713, -1.0105178, 1.1361258, -1.7399300, 1.7247891
6: -0.4965411, 1.4581203, -1.0676947, 1.6187458, -2.1152868, 2.5258150
7: -0.5581359, 0.6260951, -0.9711751, 0.9359220, -1.4940579, 1.5972703
8: -0.5503402, 0.6266192, -0.8822247, 1.1008432, -1.6511834, 1.5088439
9: -0.4185991, 0.4651394, -0.7688699, 0.8852291, -1.3038282, 1.2340094

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8817539, upper bound: 1.8464674
time: 1.68 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8826492, upper bound: 1.8482865
time: 1.72 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.4283954, 1.2709696, -0.7844378, 1.6371753, -2.0655708, 2.0554075
1: -0.5065504, 0.5120057, -0.7545584, 0.7572792, -1.2638296, 1.2665641
2: -0.5936686, 0.6343641, -0.8325227, 0.9933141, -1.5869827, 1.4668869
3: -0.4099944, 0.4626325, -0.6313741, 0.7623650, -1.1723595, 1.0940067
4: -0.5260350, 0.5839709, -0.7793881, 1.0365195, -1.5625546, 1.3633590
5: -0.6253715, 0.7425824, -0.9667646, 1.1034745, -1.7288460, 1.7093470
6: -0.5344308, 1.4685735, -1.0031563, 1.5844967, -2.1189275, 2.4717298
7: -0.5838713, 0.6439485, -0.9345270, 0.9043884, -1.4882597, 1.5784755
8: -0.5708863, 0.6576684, -0.8537888, 1.0604813, -1.6313677, 1.5114572
9: -0.4413340, 0.4900914, -0.7392236, 0.8510032, -1.2923373, 1.2293150

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8850170, upper bound: 1.8480955
time: 2.47 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8850170, upper bound: 1.8494081
time: 1.93 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.3655915, 1.1662056, -0.8663305, 1.7437718, -2.1093633, 2.0325360
1: -0.4462956, 0.4629569, -0.8130708, 0.8074290, -1.2537246, 1.2760277
2: -0.5359937, 0.5579149, -0.8903645, 1.0682744, -1.6042681, 1.4482794
3: -0.3671761, 0.3959527, -0.6797521, 0.8258173, -1.1929934, 1.0757048
4: -0.4637251, 0.4985328, -0.8397647, 1.1247712, -1.5884964, 1.3382975
5: -0.5602239, 0.6653486, -1.0506577, 1.1777655, -1.7379894, 1.7160063
6: -0.4131463, 1.4223435, -1.1253142, 1.6426642, -2.0558105, 2.5476577
7: -0.5091316, 0.5877320, -1.0080954, 0.9664483, -1.4755800, 1.5958273
8: -0.5045810, 0.5750371, -0.9167264, 1.1446393, -1.6492202, 1.4917636
9: -0.3748845, 0.4192511, -0.7970768, 0.9225988, -1.2974833, 1.2163279

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8757061, upper bound: 1.8238978
time: 2.02 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8774045, upper bound: 1.8248551
time: 2.06 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.3830998, 1.1939206, -0.7919115, 1.6434530, -2.0265527, 1.9858321
1: -0.4624637, 0.4786091, -0.7598349, 0.7628895, -1.2253532, 1.2384440
2: -0.5521010, 0.5773075, -0.8372648, 1.0011079, -1.5532088, 1.4145722
3: -0.3769137, 0.4171165, -0.6400046, 0.7686682, -1.1455818, 1.0571210
4: -0.4822236, 0.5206205, -0.7842556, 1.0456074, -1.5278311, 1.3048761
5: -0.5755689, 0.6879572, -0.9735982, 1.1139143, -1.6894832, 1.6615553
6: -0.4452221, 1.4325594, -1.0126145, 1.5860882, -2.0313103, 2.4451737
7: -0.5307168, 0.6035373, -0.9417365, 0.9100840, -1.4408008, 1.5452738
8: -0.5228423, 0.5970218, -0.8621641, 1.0699773, -1.5928197, 1.4591858
9: -0.3955613, 0.4396504, -0.7442783, 0.8592910, -1.2548523, 1.1839287

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8838231, upper bound: 1.8309926
time: 2.74 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8843870, upper bound: 1.8319511
time: 2.32 seconds

## BFS IS instance: IS_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.8261355, 1.6957741, -0.4067150, 1.2407022, -2.0668375, 2.1024890
1: -0.7845350, 0.7813374, -0.4869262, 0.4960527, -1.2805877, 1.2682636
2: -0.8624935, 1.0301049, -0.5755167, 0.6072444, -1.4697378, 1.6056216
3: -0.6554159, 0.7940063, -0.3936818, 0.4409779, -1.0963938, 1.1876881
4: -0.8107981, 1.0799718, -0.5070575, 0.5511866, -1.3619847, 1.5870293
5: -1.0105178, 1.1361258, -0.6038042, 0.7142713, -1.7247891, 1.7399300
6: -1.0676947, 1.6187458, -0.4965411, 1.4581203, -2.5258150, 2.1152868
7: -0.9711751, 0.9359220, -0.5581359, 0.6260951, -1.5972703, 1.4940579
8: -0.8822247, 1.1008432, -0.5503402, 0.6266192, -1.5088439, 1.6511834
9: -0.7688699, 0.8852291, -0.4185991, 0.4651394, -1.2340094, 1.3038282

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8464674, upper bound: 1.8817539
time: 2.10 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8482865, upper bound: 1.8826493
time: 2.25 seconds

## BFS IS instance: IS_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.7844378, 1.6371753, -0.4283954, 1.2709696, -2.0554075, 2.0655708
1: -0.7545584, 0.7572792, -0.5065504, 0.5120057, -1.2665641, 1.2638296
2: -0.8325227, 0.9933141, -0.5936686, 0.6343641, -1.4668869, 1.5869827
3: -0.6313741, 0.7623650, -0.4099944, 0.4626325, -1.0940067, 1.1723595
4: -0.7793881, 1.0365195, -0.5260350, 0.5839709, -1.3633590, 1.5625546
5: -0.9667646, 1.1034745, -0.6253715, 0.7425824, -1.7093470, 1.7288460
6: -1.0031563, 1.5844967, -0.5344308, 1.4685735, -2.4717298, 2.1189275
7: -0.9345270, 0.9043884, -0.5838713, 0.6439485, -1.5784755, 1.4882597
8: -0.8537888, 1.0604813, -0.5708863, 0.6576684, -1.5114572, 1.6313677
9: -0.7392236, 0.8510032, -0.4413340, 0.4900914, -1.2293150, 1.2923373

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8480956, upper bound: 1.8850170
time: 1.92 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8480956, upper bound: 1.8861909
time: 1.93 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.02 seconds
IS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8991594, upper bound: 1.8900493
IS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8991594, upper bound: 1.8900493
IS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8991593, upper bound: 1.8905728
IS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8991593, upper bound: 1.8905728
IS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8904415, upper bound: 1.8899819
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8904415, upper bound: 1.8899819
IS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8905455, upper bound: 1.8905455
IS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8905455, upper bound: 1.8905455
IS_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8817539, upper bound: 1.8464674
IS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8826492, upper bound: 1.8482865
IS_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8850170, upper bound: 1.8480955
IS_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8850170, upper bound: 1.8494081
IS_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8757061, upper bound: 1.8238978
IS_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8774045, upper bound: 1.8248551
IS_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8838231, upper bound: 1.8309926
IS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8843870, upper bound: 1.8319511
IS_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8464674, upper bound: 1.8817539
IS_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8482865, upper bound: 1.8826493
IS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8480956, upper bound: 1.8850170
IS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 6, lower bound: -1.8480956, upper bound: 1.8861909

## BFS IS instance: IS_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2154518, 0.9344941, -0.2195067, 0.9101569, -1.1256087, 1.1540008
1: -0.3147149, 0.3367910, -0.3105975, 0.3439346, -0.6586496, 0.6473886
2: -0.4023729, 0.4023019, -0.3956183, 0.4094609, -0.8118337, 0.7979202
3: -0.2816463, 0.2394949, -0.2785031, 0.2473902, -0.5290365, 0.5179981
4: -0.3105045, 0.3721581, -0.3083312, 0.3786796, -0.6891841, 0.6804893
5: -0.4456833, 0.4830772, -0.4361588, 0.4939380, -0.9396213, 0.9192360
6: -0.1440101, 1.3136120, -0.1254837, 1.2911390, -1.4351491, 1.4390957
7: -0.3404821, 0.4518234, -0.3465376, 0.4494008, -0.7898828, 0.7983610
8: -0.3618510, 0.3986605, -0.3604755, 0.4070444, -0.7688954, 0.7591361
9: -0.2250141, 0.2665265, -0.2356302, 0.2797877, -0.5048018, 0.5021566

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8984203, upper bound: 1.8892414
time: 2.25 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8991594, upper bound: 1.8900493
time: 2.05 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2154518, 0.9344941, -0.2219704, 0.9041200, -1.1195718, 1.1564646
1: -0.3147149, 0.3367910, -0.3107549, 0.3473970, -0.6621119, 0.6475459
2: -0.4023729, 0.4023019, -0.3946554, 0.4134376, -0.8158104, 0.7969573
3: -0.2816463, 0.2394949, -0.2784634, 0.2512371, -0.5328834, 0.5179583
4: -0.3105045, 0.3721581, -0.3088531, 0.3823284, -0.6928329, 0.6810112
5: -0.4456833, 0.4830772, -0.4338923, 0.5003511, -0.9460344, 0.9169695
6: -0.1440101, 1.3136120, -0.1225921, 1.2902631, -1.4342731, 1.4362041
7: -0.3404821, 0.4518234, -0.3501630, 0.4496260, -0.7901081, 0.8019864
8: -0.3618510, 0.3986605, -0.3643537, 0.4120111, -0.7738621, 0.7630143
9: -0.2250141, 0.2665265, -0.2397957, 0.2859881, -0.5110022, 0.5063221

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8984203, upper bound: 1.8892414
time: 2.19 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8991594, upper bound: 1.8900493
time: 2.25 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2072586, 0.8959078, -0.2336920, 0.9276679, -1.1349264, 1.1295998
1: -0.3009212, 0.3340663, -0.3219900, 0.3554728, -0.6563940, 0.6560563
2: -0.3868112, 0.3966483, -0.4061271, 0.4243887, -0.8111999, 0.8027753
3: -0.2720728, 0.2352810, -0.2861868, 0.2612684, -0.5333412, 0.5214677
4: -0.2967181, 0.3695163, -0.3218213, 0.3896247, -0.6863428, 0.6913375
5: -0.4288297, 0.4805791, -0.4451328, 0.5103453, -0.9391750, 0.9257119
6: -0.1073784, 1.2879556, -0.1471975, 1.2969497, -1.4043281, 1.4351532
7: -0.3334463, 0.4387536, -0.3616745, 0.4618227, -0.7952690, 0.8004281
8: -0.3504367, 0.3934267, -0.3729174, 0.4232192, -0.7736559, 0.7663442
9: -0.2233384, 0.2670660, -0.2493206, 0.2945979, -0.5179363, 0.5163866

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8984203, upper bound: 1.8898142
time: 1.93 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8991593, upper bound: 1.8905728
time: 2.16 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2072586, 0.8959078, -0.2360516, 0.9213381, -1.1285968, 1.1319594
1: -0.3009212, 0.3340663, -0.3221140, 0.3588542, -0.6597754, 0.6561803
2: -0.3868112, 0.3966483, -0.4051805, 0.4282180, -0.8150292, 0.8018287
3: -0.2720728, 0.2352810, -0.2861544, 0.2650071, -0.5370799, 0.5214354
4: -0.2967181, 0.3695163, -0.3222730, 0.3931868, -0.6899049, 0.6917893
5: -0.4288297, 0.4805791, -0.4428855, 0.5166113, -0.9454410, 0.9234646
6: -0.1073784, 1.2879556, -0.1440104, 1.2948549, -1.4022334, 1.4319661
7: -0.3334463, 0.4387536, -0.3651637, 0.4619819, -0.7954282, 0.8039173
8: -0.3504367, 0.3934267, -0.3766791, 0.4280382, -0.7784748, 0.7701058
9: -0.2233384, 0.2670660, -0.2533768, 0.3006897, -0.5240281, 0.5204428

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8984203, upper bound: 1.8898142
time: 2.15 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8991593, upper bound: 1.8905728
time: 2.38 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2265769, 0.9415536, -0.2195067, 0.9101569, -1.1367338, 1.1610602
1: -0.3216501, 0.3471616, -0.3105975, 0.3439346, -0.6655847, 0.6577591
2: -0.4083511, 0.4149002, -0.3956183, 0.4094609, -0.8178120, 0.8105185
3: -0.2864379, 0.2512243, -0.2785031, 0.2473902, -0.5338281, 0.5297275
4: -0.3193513, 0.3821320, -0.3083312, 0.3786796, -0.6980309, 0.6904632
5: -0.4501562, 0.4991343, -0.4361588, 0.4939380, -0.9440942, 0.9352931
6: -0.1542069, 1.3141832, -0.1254837, 1.2911390, -1.4453459, 1.4396670
7: -0.3526507, 0.4599188, -0.3465376, 0.4494008, -0.8020514, 0.8064564
8: -0.3686532, 0.4127948, -0.3604755, 0.4070444, -0.7756975, 0.7732704
9: -0.2368645, 0.2806606, -0.2356302, 0.2797877, -0.5166522, 0.5162908

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8899036, upper bound: 1.8891703
time: 2.11 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8904415, upper bound: 1.8899819
time: 2.37 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2265769, 0.9415536, -0.2219704, 0.9041200, -1.1306969, 1.1635240
1: -0.3216501, 0.3471616, -0.3107549, 0.3473970, -0.6690471, 0.6579164
2: -0.4083511, 0.4149002, -0.3946554, 0.4134376, -0.8217887, 0.8095556
3: -0.2864379, 0.2512243, -0.2784634, 0.2512371, -0.5376750, 0.5296877
4: -0.3193513, 0.3821320, -0.3088531, 0.3823284, -0.7016798, 0.6909851
5: -0.4501562, 0.4991343, -0.4338923, 0.5003511, -0.9505073, 0.9330266
6: -0.1542069, 1.3141832, -0.1225921, 1.2902631, -1.4444699, 1.4367753
7: -0.3526507, 0.4599188, -0.3501630, 0.4496260, -0.8022766, 0.8100818
8: -0.3686532, 0.4127948, -0.3643537, 0.4120111, -0.7806643, 0.7771485
9: -0.2368645, 0.2806606, -0.2397957, 0.2859881, -0.5228525, 0.5204563

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8899036, upper bound: 1.8891703
time: 2.30 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8904415, upper bound: 1.8899819
time: 7.18 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2095052, 0.8895733, -0.2336920, 0.9276679, -1.1371731, 1.1232653
1: -0.3008616, 0.3373545, -0.3219900, 0.3554728, -0.6563344, 0.6593445
2: -0.3856023, 0.4003984, -0.4061271, 0.4243887, -0.8099910, 0.8065255
3: -0.2718897, 0.2389219, -0.2861868, 0.2612684, -0.5331581, 0.5251086
4: -0.2969734, 0.3729972, -0.3218213, 0.3896247, -0.6865982, 0.6948184
5: -0.4263875, 0.4867312, -0.4451328, 0.5103453, -0.9367328, 0.9318640
6: -0.1041598, 1.2882056, -0.1471975, 1.2969497, -1.4011096, 1.4354031
7: -0.3368429, 0.4387551, -0.3616745, 0.4618227, -0.7986656, 0.8004296
8: -0.3541161, 0.3981440, -0.3729174, 0.4232192, -0.7773353, 0.7710614
9: -0.2273098, 0.2730464, -0.2493206, 0.2945979, -0.5219077, 0.5223670

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8897896, upper bound: 1.8900230
time: 2.02 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8905455, upper bound: 1.8905455
time: 2.24 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2095052, 0.8895733, -0.2360516, 0.9213381, -1.1308433, 1.1256249
1: -0.3008616, 0.3373545, -0.3221140, 0.3588542, -0.6597159, 0.6594685
2: -0.3856023, 0.4003984, -0.4051805, 0.4282180, -0.8138203, 0.8055789
3: -0.2718897, 0.2389219, -0.2861544, 0.2650071, -0.5368969, 0.5250763
4: -0.2969734, 0.3729972, -0.3222730, 0.3931868, -0.6901603, 0.6952702
5: -0.4263875, 0.4867312, -0.4428855, 0.5166113, -0.9429988, 0.9296167
6: -0.1041598, 1.2882056, -0.1440104, 1.2948549, -1.3990147, 1.4322160
7: -0.3368429, 0.4387551, -0.3651637, 0.4619819, -0.7988247, 0.8039187
8: -0.3541161, 0.3981440, -0.3766791, 0.4280382, -0.7821542, 0.7748231
9: -0.2273098, 0.2730464, -0.2533768, 0.3006897, -0.5279995, 0.5264233

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8812708, upper bound: 1.8546285
time: 1.99 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8544282, upper bound: 1.8544297
time: 1.80 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3845735, 1.2358046, -0.8010634, 1.6696776, -2.0542512, 2.0368681
1: -0.4730092, 0.4744266, -0.7670188, 0.7641536, -1.2371628, 1.2414454
2: -0.5665138, 0.5742417, -0.8456585, 1.0051625, -1.5716763, 1.4199002
3: -0.3856539, 0.4120654, -0.6399140, 0.7732075, -1.1588614, 1.0519795
4: -0.4921967, 0.5115112, -0.7929307, 1.0491253, -1.5413220, 1.3044419
5: -0.5897374, 0.6802312, -0.9862207, 1.1102405, -1.6999779, 1.6664519
6: -0.4816678, 1.4814484, -1.0347718, 1.6094160, -2.0910838, 2.5162201
7: -0.5291743, 0.6134413, -0.9469694, 0.9174546, -1.4466289, 1.5604107
8: -0.5397607, 0.5925994, -0.8610761, 1.0727658, -1.6125265, 1.4536755
9: -0.3872813, 0.4283839, -0.7488223, 0.8602611, -1.2475424, 1.1772063

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8817539, upper bound: 1.8464674
time: 1.53 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8817539, upper bound: 1.8464674
time: 1.58 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3759446, 1.1946460, -0.8261355, 1.6957741, -2.0717187, 2.0207815
1: -0.4582976, 0.4706964, -0.7845350, 0.7813374, -1.2396350, 1.2552314
2: -0.5494688, 0.5677172, -0.8624935, 1.0301049, -1.5795736, 1.4302106
3: -0.3752226, 0.4059608, -0.6554159, 0.7940063, -1.1692290, 1.0613767
4: -0.4767937, 0.5077250, -0.8107981, 1.0799718, -1.5567656, 1.3185232
5: -0.5728263, 0.6767382, -1.0105178, 1.1361258, -1.7089521, 1.6872561
6: -0.4423463, 1.4436779, -1.0676947, 1.6187458, -2.0610921, 2.5113726
7: -0.5206935, 0.5994501, -0.9711751, 0.9359220, -1.4566156, 1.5706253
8: -0.5195796, 0.5863240, -0.8822247, 1.1008432, -1.6204228, 1.4685487
9: -0.3831031, 0.4269175, -0.7688699, 0.8852291, -1.2683322, 1.1957874

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8437101, upper bound: 1.8335032
time: 1.90 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8324977, upper bound: 1.7887492
time: 1.86 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4359326, 1.3020326, -0.7844378, 1.6371753, -2.0731080, 2.0864704
1: -0.5178497, 0.5148485, -0.7545584, 0.7572792, -1.2751288, 1.2694068
2: -0.6063541, 0.6408684, -0.8325227, 0.9933141, -1.5996681, 1.4733912
3: -0.4187015, 0.4680504, -0.6313741, 0.7623650, -1.1810665, 1.0994245
4: -0.5376877, 0.5897490, -0.7793881, 1.0365195, -1.5742072, 1.3691370
5: -0.6389771, 0.7447645, -0.9667646, 1.1034745, -1.7424517, 1.7115290
6: -0.5636270, 1.4948835, -1.0031563, 1.5844967, -2.1481237, 2.4980397
7: -0.5916014, 0.6543775, -0.9345270, 0.9043884, -1.4959898, 1.5889045
8: -0.5861527, 0.6635457, -0.8537888, 1.0604813, -1.6466340, 1.5173345
9: -0.4466369, 0.4923048, -0.7392236, 0.8510032, -1.2976402, 1.2315284

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_B1_B2_A1_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8823345, upper bound: 1.8480955
time: 1.82 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_A2

### Relational analysis result of IS_A1_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8823345, upper bound: 1.8480956
time: 2.16 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3920232, 1.2182788, -0.7844378, 1.6371753, -2.0291986, 2.0027165
1: -0.4730298, 0.4845930, -0.7545584, 0.7572792, -1.2303089, 1.2391515
2: -0.5629200, 0.5874848, -0.8325227, 0.9933141, -1.5562341, 1.4200075
3: -0.3839064, 0.4251538, -0.6313741, 0.7623650, -1.1462715, 1.0565279
4: -0.4930536, 0.5296794, -0.7793881, 1.0365195, -1.5295732, 1.3090675
5: -0.5880038, 0.6964785, -0.9667646, 1.1034745, -1.6914783, 1.6632431
6: -0.4697315, 1.4505521, -1.0031563, 1.5844967, -2.0542283, 2.4537084
7: -0.5404966, 0.6134367, -0.9345270, 0.9043884, -1.4448850, 1.5479637
8: -0.5357551, 0.6065232, -0.8537888, 1.0604813, -1.5962365, 1.4603119
9: -0.4024153, 0.4471083, -0.7392236, 0.8510032, -1.2534184, 1.1863319

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_B1_B2_A2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8811585, upper bound: 1.8478598
time: 1.83 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8823343, upper bound: 1.8494081
time: 2.33 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3485748, 1.1677868, -0.8417909, 1.7182316, -2.0668063, 2.0095778
1: -0.4381046, 0.4421660, -0.7959467, 0.7905342, -1.2286389, 1.2381127
2: -0.5283650, 0.5374160, -0.8738678, 1.0438626, -1.5722275, 1.4112837
3: -0.3639123, 0.3696500, -0.6645877, 0.8054413, -1.1693537, 1.0342376
4: -0.4517068, 0.4714861, -0.8222656, 1.0944622, -1.5461690, 1.2937517
5: -0.5577434, 0.6340344, -1.0268850, 1.1523972, -1.7101405, 1.6609194
6: -0.4051875, 1.4464309, -1.0930567, 1.6335671, -2.0387545, 2.5394876
7: -0.4842690, 0.5795473, -0.9843831, 0.9482641, -1.4325331, 1.5639305
8: -0.4989597, 0.5488747, -0.8960057, 1.1170306, -1.6159903, 1.4448805
9: -0.3478607, 0.3918377, -0.7774466, 0.8980604, -1.2459211, 1.1692842

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8749391, upper bound: 1.8238898
time: 2.06 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8755558, upper bound: 1.8238689
time: 1.94 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3393212, 1.1257526, -0.8663305, 1.7437718, -2.0830929, 1.9920831
1: -0.4229404, 0.4379553, -0.8130708, 0.8074290, -1.2303694, 1.2510262
2: -0.5106964, 0.5303582, -0.8903645, 1.0682744, -1.5789708, 1.4207227
3: -0.3531283, 0.3632110, -0.6797521, 0.8258173, -1.1789455, 1.0429631
4: -0.4358675, 0.4672106, -0.8397647, 1.1247712, -1.5606387, 1.3069754
5: -0.5401825, 0.6297803, -1.0506577, 1.1777655, -1.7179480, 1.6804380
6: -0.3648041, 1.4087173, -1.1253142, 1.6426642, -2.0074682, 2.5340314
7: -0.4752215, 0.5651297, -1.0080954, 0.9664483, -1.4416698, 1.5732250
8: -0.4783898, 0.5418918, -0.9167264, 1.1446393, -1.6230290, 1.4586182
9: -0.3435592, 0.3899766, -0.7970768, 0.9225988, -1.2661581, 1.1870534

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8356068, upper bound: 1.8138954
time: 2.30 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8196002, upper bound: 1.7537136
time: 1.92 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3649022, 1.1936924, -0.7672825, 1.6179278, -1.9828299, 1.9609749
1: -0.4526561, 0.4575639, -0.7425915, 0.7459769, -1.1986330, 1.2001554
2: -0.5441629, 0.5545390, -0.8206791, 0.9766487, -1.5208116, 1.3752180
3: -0.3727646, 0.3896402, -0.6218454, 0.7482450, -1.1210096, 1.0114856
4: -0.4689553, 0.4907323, -0.7666720, 1.0152607, -1.4842160, 1.2574043
5: -0.5704059, 0.6563828, -0.9496925, 1.0885332, -1.6589391, 1.6060753
6: -0.4357469, 1.4564914, -0.9801387, 1.5769963, -2.0127432, 2.4366300
7: -0.5050893, 0.5936591, -0.9179869, 0.8917904, -1.3968798, 1.5116460
8: -0.5153018, 0.5694345, -0.8414190, 1.0423281, -1.5576299, 1.4108535
9: -0.3667463, 0.4097673, -0.7246048, 0.8347075, -1.2014538, 1.1343721

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8752946, upper bound: 1.8296678
time: 2.19 seconds

## Relational analysis of IS_A1_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8752946, upper bound: 1.8309926
time: 2.23 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3558171, 1.1518829, -0.7919115, 1.6434530, -1.9992701, 1.9437944
1: -0.4377113, 0.4535436, -0.7598349, 0.7628895, -1.2006007, 1.2133784
2: -0.5267102, 0.5476529, -0.8372648, 1.0011079, -1.5278182, 1.3849177
3: -0.3621054, 0.3834198, -0.6400046, 0.7686682, -1.1307735, 1.0234245
4: -0.4533267, 0.4866716, -0.7842556, 1.0456074, -1.4989341, 1.2709272
5: -0.5530689, 0.6523601, -0.9735982, 1.1139143, -1.6669831, 1.6259582
6: -0.3956128, 1.4185327, -1.0126145, 1.5860882, -1.9817009, 2.4311471
7: -0.4962906, 0.5794485, -0.9417365, 0.9100840, -1.4063747, 1.5211849
8: -0.4950649, 0.5627614, -0.8621641, 1.0699773, -1.5650423, 1.4249256
9: -0.3626336, 0.4081816, -0.7442783, 0.8592910, -1.2219245, 1.1524599

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8312210
time: 2.06 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8319511
time: 2.37 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.8010634, 1.6696776, -0.3845735, 1.2358046, -2.0368681, 2.0542512
1: -0.7670188, 0.7641536, -0.4730092, 0.4744266, -1.2414454, 1.2371628
2: -0.8456585, 1.0051625, -0.5665138, 0.5742417, -1.4199002, 1.5716763
3: -0.6399140, 0.7732075, -0.3856539, 0.4120654, -1.0519795, 1.1588614
4: -0.7929307, 1.0491253, -0.4921967, 0.5115112, -1.3044419, 1.5413220
5: -0.9862207, 1.1102405, -0.5897374, 0.6802312, -1.6664519, 1.6999779
6: -1.0347718, 1.6094160, -0.4816678, 1.4814484, -2.5162201, 2.0910838
7: -0.9469694, 0.9174546, -0.5291743, 0.6134413, -1.5604107, 1.4466289
8: -0.8610761, 1.0727658, -0.5397607, 0.5925994, -1.4536755, 1.6125265
9: -0.7488223, 0.8602611, -0.3872813, 0.4283839, -1.1772063, 1.2475424

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8464674, upper bound: 1.8817539
time: 2.01 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8464674, upper bound: 1.8817539
time: 2.02 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.8261355, 1.6957741, -0.3759446, 1.1946460, -2.0207815, 2.0717187
1: -0.7845350, 0.7813374, -0.4582976, 0.4706964, -1.2552314, 1.2396350
2: -0.8624935, 1.0301049, -0.5494688, 0.5677172, -1.4302106, 1.5795736
3: -0.6554159, 0.7940063, -0.3752226, 0.4059608, -1.0613767, 1.1692290
4: -0.8107981, 1.0799718, -0.4767937, 0.5077250, -1.3185232, 1.5567656
5: -1.0105178, 1.1361258, -0.5728263, 0.6767382, -1.6872561, 1.7089521
6: -1.0676947, 1.6187458, -0.4423463, 1.4436779, -2.5113726, 2.0610921
7: -0.9711751, 0.9359220, -0.5206935, 0.5994501, -1.5706253, 1.4566156
8: -0.8822247, 1.1008432, -0.5195796, 0.5863240, -1.4685487, 1.6204228
9: -0.7688699, 0.8852291, -0.3831031, 0.4269175, -1.1957874, 1.2683322

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8335032, upper bound: 1.8437101
time: 2.16 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7887492, upper bound: 1.8324977
time: 1.72 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.7844378, 1.6371753, -0.4359326, 1.3020326, -2.0864704, 2.0731080
1: -0.7545584, 0.7572792, -0.5178497, 0.5148485, -1.2694068, 1.2751288
2: -0.8325227, 0.9933141, -0.6063541, 0.6408684, -1.4733912, 1.5996681
3: -0.6313741, 0.7623650, -0.4187015, 0.4680504, -1.0994245, 1.1810665
4: -0.7793881, 1.0365195, -0.5376877, 0.5897490, -1.3691370, 1.5742072
5: -0.9667646, 1.1034745, -0.6389771, 0.7447645, -1.7115290, 1.7424517
6: -1.0031563, 1.5844967, -0.5636270, 1.4948835, -2.4980397, 2.1481237
7: -0.9345270, 0.9043884, -0.5916014, 0.6543775, -1.5889045, 1.4959898
8: -0.8537888, 1.0604813, -0.5861527, 0.6635457, -1.5173345, 1.6466340
9: -0.7392236, 0.8510032, -0.4466369, 0.4923048, -1.2315284, 1.2976402

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8480956, upper bound: 1.8850170
time: 2.12 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8480956, upper bound: 1.8850170
time: 2.00 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.7844378, 1.6371753, -0.3920232, 1.2182788, -2.0027165, 2.0291986
1: -0.7545584, 0.7572792, -0.4730298, 0.4845930, -1.2391515, 1.2303089
2: -0.8325227, 0.9933141, -0.5629200, 0.5874848, -1.4200075, 1.5562341
3: -0.6313741, 0.7623650, -0.3839064, 0.4251538, -1.0565279, 1.1462715
4: -0.7793881, 1.0365195, -0.4930536, 0.5296794, -1.3090675, 1.5295732
5: -0.9667646, 1.1034745, -0.5880038, 0.6964785, -1.6632431, 1.6914783
6: -1.0031563, 1.5844967, -0.4697315, 1.4505521, -2.4537084, 2.0542283
7: -0.9345270, 0.9043884, -0.5404966, 0.6134367, -1.5479637, 1.4448850
8: -0.8537888, 1.0604813, -0.5357551, 0.6065232, -1.4603119, 1.5962365
9: -0.7392236, 0.8510032, -0.4024153, 0.4471083, -1.1863319, 1.2534184

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8443778, upper bound: 1.8858963
time: 2.20 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8480956, upper bound: 1.8861910
time: 2.52 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.89 seconds
IS_A1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8984203, upper bound: 1.8892414
IS_A1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8991594, upper bound: 1.8900493
IS_A1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8984203, upper bound: 1.8892414
IS_A1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8991594, upper bound: 1.8900493
IS_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8984203, upper bound: 1.8898142
IS_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8991593, upper bound: 1.8905728
IS_A1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8984203, upper bound: 1.8898142
IS_A1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8991593, upper bound: 1.8905728
IS_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8899036, upper bound: 1.8891703
IS_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8904415, upper bound: 1.8899819
IS_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8899036, upper bound: 1.8891703
IS_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8904415, upper bound: 1.8899819
IS_A1_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8897896, upper bound: 1.8900230
IS_A1_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8905455, upper bound: 1.8905455
IS_A1_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8812708, upper bound: 1.8546285
IS_A1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8544282, upper bound: 1.8544297
IS_A1_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8817539, upper bound: 1.8464674
IS_A1_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8817539, upper bound: 1.8464674
IS_A1_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8437101, upper bound: 1.8335032
IS_A1_B2_B1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8324977, upper bound: 1.7887492
IS_A1_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8823345, upper bound: 1.8480955
IS_A1_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8823345, upper bound: 1.8480956
IS_A1_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8811585, upper bound: 1.8478598
IS_A1_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8823343, upper bound: 1.8494081
IS_A1_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8749391, upper bound: 1.8238898
IS_A1_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8755558, upper bound: 1.8238689
IS_A1_B2_B2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8356068, upper bound: 1.8138954
IS_A1_B2_B2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8196002, upper bound: 1.7537136
IS_A1_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8752946, upper bound: 1.8296678
IS_A1_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8752946, upper bound: 1.8309926
IS_A1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8312210
IS_A1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8773133, upper bound: 1.8319511
IS_A2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8464674, upper bound: 1.8817539
IS_A2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8464674, upper bound: 1.8817539
IS_A2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8335032, upper bound: 1.8437101
IS_A2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.7887492, upper bound: 1.8324977
IS_A2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8480956, upper bound: 1.8850170
IS_A2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8480956, upper bound: 1.8850170
IS_A2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8443778, upper bound: 1.8858963
IS_A2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.89
Output dim: 6, lower bound: -1.8480956, upper bound: 1.8861910

## BFS IS instance: IS_A1_B1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.1992713, 0.9146871, -0.2240654, 0.9332255, -1.1324967, 1.1387525
1: -0.3016900, 0.3237556, -0.3184277, 0.3449340, -0.6466240, 0.6421833
2: -0.3902865, 0.3853621, -0.4045457, 0.4120911, -0.8023776, 0.7899078
3: -0.2729115, 0.2237414, -0.2840719, 0.2497660, -0.5226775, 0.5078133
4: -0.2951659, 0.3597881, -0.3166633, 0.3790966, -0.6742625, 0.6764514
5: -0.4355428, 0.4644218, -0.4459834, 0.4924448, -0.9279876, 0.9104052
6: -0.1188300, 1.3053600, -0.1463477, 1.3040482, -1.4228783, 1.4517076
7: -0.3232574, 0.4377548, -0.3504135, 0.4569526, -0.7802099, 0.7881683
8: -0.3480145, 0.3803610, -0.3645772, 0.4085877, -0.7566022, 0.7449383
9: -0.2095654, 0.2498562, -0.2373872, 0.2784784, -0.4880438, 0.4872434

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8882222, upper bound: 1.8621048
time: 2.40 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8618688, upper bound: 1.8617999
time: 1.77 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.2154518, 0.9344941, -0.1910335, 0.8744125, -1.0898643, 1.1255276
1: -0.3147149, 0.3367910, -0.2874907, 0.3210520, -0.6357669, 0.6242817
2: -0.4023729, 0.4023019, -0.3742104, 0.3797177, -0.7820905, 0.7765123
3: -0.2816463, 0.2394949, -0.2629691, 0.2196814, -0.5013278, 0.5024641
4: -0.3105045, 0.3721581, -0.2811005, 0.3570620, -0.6675665, 0.6532586
5: -0.4456833, 0.4830772, -0.4179860, 0.4617836, -0.9074669, 0.9010632
6: -0.1440101, 1.3136120, -0.0810578, 1.2809436, -1.4249537, 1.3946698
7: -0.3404821, 0.4518234, -0.3162356, 0.4242360, -0.7647181, 0.7680590
8: -0.3618510, 0.3986605, -0.3359746, 0.3749803, -0.7368313, 0.7346351
9: -0.2250141, 0.2665265, -0.2082598, 0.2505935, -0.4756077, 0.4747863

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8899332, upper bound: 1.8627186
time: 1.97 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8622863, upper bound: 1.8624052
time: 2.03 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.1992713, 0.9146871, -0.2430249, 0.9576500, -1.1569213, 1.1577120
1: -0.3016900, 0.3237556, -0.3340049, 0.3605127, -0.6622027, 0.6577605
2: -0.3902865, 0.3853621, -0.4189556, 0.4324076, -0.8226941, 0.8043178
3: -0.2729115, 0.2237414, -0.2943027, 0.2680599, -0.5409714, 0.5180441
4: -0.2951659, 0.3597881, -0.3343638, 0.3944111, -0.6895770, 0.6941519
5: -0.4355428, 0.4644218, -0.4582541, 0.5166931, -0.9522359, 0.9226758
6: -0.1188300, 1.3053600, -0.1768547, 1.3163049, -1.4351349, 1.4822147
7: -0.3232574, 0.4377548, -0.3706639, 0.4735220, -0.7967794, 0.8084186
8: -0.3480145, 0.3803610, -0.3809146, 0.4310794, -0.7790939, 0.7612756
9: -0.2095654, 0.2498562, -0.2544016, 0.2984248, -0.5079901, 0.5042577

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8843058, upper bound: 1.8495865
time: 1.87 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8609583, upper bound: 1.8492783
time: 1.94 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.2154518, 0.9344941, -0.1928261, 0.8672711, -1.0827229, 1.1273203
1: -0.3147149, 0.3367910, -0.2870591, 0.3239554, -0.6386703, 0.6238501
2: -0.4023729, 0.4023019, -0.3726139, 0.3829908, -0.7853637, 0.7749158
3: -0.2816463, 0.2394949, -0.2624988, 0.2228808, -0.5045271, 0.5019938
4: -0.3105045, 0.3721581, -0.2809156, 0.3601927, -0.6706972, 0.6530737
5: -0.4456833, 0.4830772, -0.4151758, 0.4674203, -0.9131036, 0.8982530
6: -0.1440101, 1.3136120, -0.0771782, 1.2824421, -1.4264522, 1.3907902
7: -0.3404821, 0.4518234, -0.3191629, 0.4238093, -0.7642914, 0.7709863
8: -0.3618510, 0.3986605, -0.3392406, 0.3791988, -0.7410498, 0.7379012
9: -0.2250141, 0.2665265, -0.2118111, 0.2561131, -0.4811273, 0.4783376

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8884776, upper bound: 1.8541485
time: 2.23 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8620255, upper bound: 1.8538202
time: 2.27 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.1908983, 0.8756229, -0.2382691, 0.9507946, -1.1416929, 1.1138921
1: -0.2876973, 0.3208417, -0.3297816, 0.3564569, -0.6441542, 0.6506233
2: -0.3745519, 0.3795155, -0.4150444, 0.4270655, -0.8016174, 0.7945598
3: -0.2631456, 0.2193505, -0.2916944, 0.2636827, -0.5268283, 0.5110449
4: -0.2811765, 0.3569371, -0.3300874, 0.3899961, -0.6711726, 0.6870245
5: -0.4184634, 0.4617238, -0.4548751, 0.5089113, -0.9273747, 0.9165989
6: -0.0819756, 1.2819579, -0.1680785, 1.3109281, -1.3929037, 1.4500364
7: -0.3160120, 0.4243731, -0.3655823, 0.4693472, -0.7853591, 0.7899554
8: -0.3360884, 0.3748245, -0.3765907, 0.4246833, -0.7607718, 0.7514152
9: -0.2076989, 0.2501320, -0.2510976, 0.2932430, -0.5009419, 0.5012295

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8628802, upper bound: 1.8899332
time: 2.44 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8626091, upper bound: 1.8623924
time: 1.99 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.2072586, 0.8959078, -0.2051244, 0.8915004, -1.0987589, 1.1010323
1: -0.3009212, 0.3340663, -0.2987980, 0.3325009, -0.6334221, 0.6328643
2: -0.3868112, 0.3966483, -0.3846641, 0.3945331, -0.7813443, 0.7813123
3: -0.2720728, 0.2352810, -0.2706087, 0.2334781, -0.5055509, 0.5058897
4: -0.2967181, 0.3695163, -0.2944325, 0.3679506, -0.6646687, 0.6639488
5: -0.4288297, 0.4805791, -0.4267558, 0.4781152, -0.9069449, 0.9073349
6: -0.1073784, 1.2879556, -0.1026776, 1.2858521, -1.3932304, 1.3906333
7: -0.3334463, 0.4387536, -0.3313026, 0.4365475, -0.7699938, 0.7700561
8: -0.3504367, 0.3934267, -0.3483621, 0.3910702, -0.7415069, 0.7417889
9: -0.2233384, 0.2670660, -0.2218433, 0.2653263, -0.4886647, 0.4889093

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8632006, upper bound: 1.8899332
time: 2.02 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8629862, upper bound: 1.8629863
time: 1.92 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.1908983, 0.8756229, -0.2565551, 0.9744402, -1.1653385, 1.1321781
1: -0.2876973, 0.3208417, -0.3448424, 0.3715456, -0.6592429, 0.6656842
2: -0.3745519, 0.3795155, -0.4289511, 0.4466715, -0.8212233, 0.8084666
3: -0.2631456, 0.2193505, -0.3015637, 0.2813208, -0.5444664, 0.5209142
4: -0.2811765, 0.3569371, -0.3472072, 0.4048828, -0.6860592, 0.7041442
5: -0.4184634, 0.4617238, -0.4669372, 0.5324135, -0.9508769, 0.9286610
6: -0.0819756, 1.2819579, -0.1975362, 1.3229021, -1.4048777, 1.4794941
7: -0.3160120, 0.4243731, -0.3851250, 0.4853222, -0.8013341, 0.8094981
8: -0.3360884, 0.3748245, -0.3923242, 0.4465339, -0.7826223, 0.7671487
9: -0.2076989, 0.2501320, -0.2674738, 0.3125915, -0.5202905, 0.5176058

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8622780, upper bound: 1.8802745
time: 1.90 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8618168, upper bound: 1.8499085
time: 1.98 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.2072586, 0.8959078, -0.2067769, 0.8844675, -1.0917261, 1.1026847
1: -0.3009212, 0.3340663, -0.2982824, 0.3353040, -0.6362252, 0.6323487
2: -0.3868112, 0.3966483, -0.3830329, 0.3976636, -0.7844748, 0.7796812
3: -0.2720728, 0.2352810, -0.2701186, 0.2365419, -0.5086147, 0.5053996
4: -0.2967181, 0.3695163, -0.2941514, 0.3709590, -0.6676772, 0.6636676
5: -0.4288297, 0.4805791, -0.4239522, 0.4835648, -0.9123945, 0.9045314
6: -0.1073784, 1.2879556, -0.0985513, 1.2864097, -1.3937881, 1.3865070
7: -0.3334463, 0.4387536, -0.3340706, 0.4360552, -0.7695016, 0.7728242
8: -0.3504367, 0.3934267, -0.3514939, 0.3950947, -0.7455313, 0.7449206
9: -0.2233384, 0.2670660, -0.2252686, 0.2706831, -0.4940215, 0.4923346

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8884776, upper bound: 1.8546595
time: 1.89 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8627467, upper bound: 1.8544514
time: 1.83 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.2104456, 0.9213287, -0.2240654, 0.9332255, -1.1436710, 1.1453941
1: -0.3086960, 0.3340848, -0.3184277, 0.3449340, -0.6536300, 0.6525124
2: -0.3963239, 0.3979612, -0.4045457, 0.4120911, -0.8084149, 0.8025069
3: -0.2777308, 0.2354796, -0.2840719, 0.2497660, -0.5274968, 0.5195515
4: -0.3040550, 0.3697326, -0.3166633, 0.3790966, -0.6831515, 0.6863959
5: -0.4399903, 0.4805626, -0.4459834, 0.4924448, -0.9324350, 0.9265459
6: -0.1291885, 1.3064616, -0.1463477, 1.3040482, -1.4332367, 1.4528093
7: -0.3354445, 0.4457849, -0.3504135, 0.4569526, -0.7923970, 0.7961984
8: -0.3548898, 0.3944669, -0.3645772, 0.4085877, -0.7634775, 0.7590442
9: -0.2213693, 0.2639343, -0.2373872, 0.2784784, -0.4998477, 0.5013216

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8528223, upper bound: 1.8851599
time: 2.25 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8525192, upper bound: 1.8610949
time: 1.80 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.2265769, 0.9415536, -0.1910335, 0.8744125, -1.1009893, 1.1325872
1: -0.3216501, 0.3471616, -0.2874907, 0.3210520, -0.6427021, 0.6346523
2: -0.4083511, 0.4149002, -0.3742104, 0.3797177, -0.7880688, 0.7891105
3: -0.2864379, 0.2512243, -0.2629691, 0.2196814, -0.5061194, 0.5141935
4: -0.3193513, 0.3821320, -0.2811005, 0.3570620, -0.6764134, 0.6632324
5: -0.4501562, 0.4991343, -0.4179860, 0.4617836, -0.9119397, 0.9171203
6: -0.1542069, 1.3141832, -0.0810578, 1.2809436, -1.4351506, 1.3952410
7: -0.3526507, 0.4599188, -0.3162356, 0.4242360, -0.7768867, 0.7761544
8: -0.3686532, 0.4127948, -0.3359746, 0.3749803, -0.7436335, 0.7487694
9: -0.2368645, 0.2806606, -0.2082598, 0.2505935, -0.4874580, 0.4889205

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8810647, upper bound: 1.8623144
time: 1.73 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8531095, upper bound: 1.8618521
time: 1.94 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.2104456, 0.9213287, -0.2430249, 0.9576500, -1.1680956, 1.1643536
1: -0.3086960, 0.3340848, -0.3340049, 0.3605127, -0.6692086, 0.6680897
2: -0.3963239, 0.3979612, -0.4189556, 0.4324076, -0.8287314, 0.8169168
3: -0.2777308, 0.2354796, -0.2943027, 0.2680599, -0.5457908, 0.5297823
4: -0.3040550, 0.3697326, -0.3343638, 0.3944111, -0.6984661, 0.7040963
5: -0.4399903, 0.4805626, -0.4582541, 0.5166931, -0.9566834, 0.9388167
6: -0.1291885, 1.3064616, -0.1768547, 1.3163049, -1.4454935, 1.4833162
7: -0.3354445, 0.4457849, -0.3706639, 0.4735220, -0.8089665, 0.8164488
8: -0.3548898, 0.3944669, -0.3809146, 0.4310794, -0.7859692, 0.7753814
9: -0.2213693, 0.2639343, -0.2544016, 0.2984248, -0.5197941, 0.5183358

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8526566, upper bound: 1.8772503
time: 1.91 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8522712, upper bound: 1.8491903
time: 2.13 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.2265769, 0.9415536, -0.1928261, 0.8672711, -1.0938480, 1.1343797
1: -0.3216501, 0.3471616, -0.2870591, 0.3239554, -0.6456056, 0.6342206
2: -0.4083511, 0.4149002, -0.3726139, 0.3829908, -0.7913419, 0.7875141
3: -0.2864379, 0.2512243, -0.2624988, 0.2228808, -0.5093187, 0.5137231
4: -0.3193513, 0.3821320, -0.2809156, 0.3601927, -0.6795441, 0.6630476
5: -0.4501562, 0.4991343, -0.4151758, 0.4674203, -0.9175764, 0.9143101
6: -0.1542069, 1.3141832, -0.0771782, 1.2824421, -1.4366491, 1.3913615
7: -0.3526507, 0.4599188, -0.3191629, 0.4238093, -0.7764599, 0.7790817
8: -0.3686532, 0.4127948, -0.3392406, 0.3791988, -0.7478520, 0.7520354
9: -0.2368645, 0.2806606, -0.2118111, 0.2561131, -0.4929776, 0.4924718

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8809846, upper bound: 1.8540616
time: 1.84 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8530874, upper bound: 1.8537294
time: 2.09 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2322737, 0.9449037, -0.2172361, 0.9069000, -1.1391737, 1.1621398
1: -0.3255079, 0.3518463, -0.3086752, 0.3421522, -0.6676601, 0.6605215
2: -0.4112139, 0.4211518, -0.3938075, 0.4071273, -0.8183412, 0.8149593
3: -0.2886755, 0.2574277, -0.2772135, 0.2452471, -0.5339226, 0.5346412
4: -0.3241682, 0.3863555, -0.3061112, 0.3769857, -0.7011540, 0.6924667
5: -0.4517915, 0.5049229, -0.4345517, 0.4914089, -0.9432003, 0.9394745
6: -0.1608663, 1.3122091, -0.1216769, 1.2899801, -1.4508463, 1.4338861
7: -0.3591601, 0.4641896, -0.3441595, 0.4473241, -0.8064842, 0.8083491
8: -0.3720658, 0.4191143, -0.3585033, 0.4045241, -0.7765899, 0.7776176
9: -0.2436304, 0.2872548, -0.2335665, 0.2775798, -0.5212103, 0.5208212

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8802745, upper bound: 1.8622780
time: 1.89 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8499085, upper bound: 1.8618168
time: 1.78 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1806060, 0.8531412, -0.2336920, 0.9276679, -1.1082739, 1.0868332
1: -0.2773540, 0.3141760, -0.3219900, 0.3554728, -0.6328268, 0.6361660
2: -0.3637846, 0.3703282, -0.4061271, 0.4243887, -0.7881733, 0.7764553
3: -0.2560428, 0.2108136, -0.2861868, 0.2612684, -0.5173112, 0.4970003
4: -0.2693838, 0.3510133, -0.3218213, 0.3896247, -0.6590086, 0.6728346
5: -0.4080175, 0.4540177, -0.4451328, 0.5103453, -0.9183627, 0.8991506
6: -0.0590689, 1.2809784, -0.1471975, 1.2969497, -1.3560185, 1.4281759
7: -0.3060597, 0.4132388, -0.3616745, 0.4618227, -0.7678825, 0.7749133
8: -0.3291885, 0.3655639, -0.3729174, 0.4232192, -0.7524077, 0.7384813
9: -0.1996495, 0.2433801, -0.2493206, 0.2945979, -0.4942474, 0.4927007

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8546595, upper bound: 1.8884776
time: 2.22 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8544515, upper bound: 1.8627467
time: 1.86 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1219662, 0.7201665, -0.2360516, 0.9213381, -1.0433043, 0.9562181
1: -0.2095885, 0.2605000, -0.3221140, 0.3588542, -0.5684428, 0.5826140
2: -0.2977007, 0.3015837, -0.4051805, 0.4282180, -0.7259187, 0.7067642
3: -0.2127289, 0.1400333, -0.2861544, 0.2650071, -0.4777360, 0.4261877
4: -0.1980772, 0.3018718, -0.3222730, 0.3931868, -0.5912640, 0.6241448
5: -0.3462314, 0.3818065, -0.4428855, 0.5166113, -0.8628427, 0.8246920
6: 0.0890885, 1.2659496, -0.1440104, 1.2948549, -1.2057664, 1.4099600
7: -0.2321927, 0.3450118, -0.3651637, 0.4619819, -0.6941746, 0.7101755
8: -0.2693418, 0.2837766, -0.3766791, 0.4280382, -0.6973799, 0.6604556
9: -0.1428810, 0.1976597, -0.2533768, 0.3006897, -0.4435707, 0.4510365

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8801938, upper bound: 1.8540221
time: 2.40 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8812708, upper bound: 1.8546285
time: 2.18 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1240438, 0.6463983, -0.1996390, 0.8670249, -0.9910687, 0.8460373
1: -0.1796974, 0.2353332, -0.2906122, 0.3306332, -0.5103306, 0.5259454
2: -0.2656757, 0.2714065, -0.3750900, 0.3910360, -0.6567117, 0.6464965
3: -0.2001190, 0.1124446, -0.2649561, 0.2308652, -0.4309843, 0.3774008
4: -0.1656332, 0.2814979, -0.2858509, 0.3664552, -0.5320885, 0.5673488
5: -0.3151710, 0.3559790, -0.4159358, 0.4769855, -0.7921565, 0.7719148
6: 0.1645296, 1.2700239, -0.0806559, 1.2839589, -1.1194293, 1.3506799
7: -0.1988565, 0.3165172, -0.3271010, 0.4282016, -0.6270581, 0.6436182
8: -0.2432683, 0.2527610, -0.3454542, 0.3881215, -0.6313899, 0.5982152
9: -0.1222818, 0.1839219, -0.2205907, 0.2661330, -0.3884148, 0.4045126

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8498948, upper bound: 1.8536285
time: 2.18 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8544282, upper bound: 1.8544297
time: 1.94 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.1992713, 0.9146871, -0.8010634, 1.6696776, -1.8689489, 1.7157505
1: -0.3016900, 0.3237556, -0.7670188, 0.7641536, -1.0658436, 1.0907744
2: -0.3902865, 0.3853621, -0.8456585, 1.0051625, -1.3954489, 1.2310207
3: -0.2729115, 0.2237414, -0.6399140, 0.7732075, -1.0461190, 0.8636554
4: -0.2951659, 0.3597881, -0.7929307, 1.0491253, -1.3442912, 1.1527188
5: -0.4355428, 0.4644218, -0.9862207, 1.1102405, -1.5457833, 1.4506426
6: -0.1188300, 1.3053600, -1.0347718, 1.6094160, -1.7282460, 2.3401318
7: -0.3232574, 0.4377548, -0.9469694, 0.9174546, -1.2407119, 1.3847241
8: -0.3480145, 0.3803610, -0.8610761, 1.0727658, -1.4207804, 1.2414372
9: -0.2095654, 0.2498562, -0.7488223, 0.8602611, -1.0698265, 0.9986785

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8392952, upper bound: 1.8199692
time: 2.25 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B2_B1_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8239831, upper bound: 1.7805475
time: 1.71 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.2102968, 0.9211587, -0.8010634, 1.6696776, -1.8799744, 1.7222221
1: -0.3085759, 0.3338820, -0.7670188, 0.7641536, -1.0727296, 1.1009008
2: -0.3962048, 0.3978045, -0.8456585, 1.0051625, -1.4013673, 1.2434629
3: -0.2776511, 0.2353318, -0.6399140, 0.7732075, -1.0508586, 0.8752458
4: -0.3039144, 0.3694724, -0.7929307, 1.0491253, -1.3530397, 1.1624031
5: -0.4399044, 0.4803667, -0.9862207, 1.1102405, -1.5501449, 1.4665874
6: -0.1289688, 1.3063754, -1.0347718, 1.6094160, -1.7383848, 2.3411472
7: -0.3352553, 0.4456540, -0.9469694, 0.9174546, -1.2527099, 1.3926234
8: -0.3547675, 0.3940904, -0.8610761, 1.0727658, -1.4275334, 1.2551665
9: -0.2212234, 0.2636225, -0.7488223, 0.8602611, -1.0814846, 1.0124449

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8392952, upper bound: 1.8199692
time: 1.85 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8239831, upper bound: 1.7805475
time: 1.51 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3759446, 1.1946460, -0.6408266, 1.4699628, -1.8459074, 1.8354726
1: -0.4582976, 0.4706964, -0.6528313, 0.6626136, -1.1209111, 1.1235278
2: -0.5494688, 0.5677172, -0.7324177, 0.8555449, -1.4050137, 1.3001349
3: -0.3752226, 0.4059608, -0.5431786, 0.6473091, -1.0225317, 0.9491394
4: -0.4767937, 0.5077250, -0.6751659, 0.8704064, -1.3472002, 1.1828909
5: -0.5728263, 0.6767382, -0.8239776, 0.9598182, -1.5326445, 1.5007159
6: -0.4423463, 1.4436779, -0.8017728, 1.5047332, -1.9470794, 2.2454507
7: -0.5206935, 0.5994501, -0.8008890, 0.7960505, -1.3167441, 1.4003391
8: -0.5195796, 0.5863240, -0.7347701, 0.9041102, -1.4236898, 1.3210940
9: -0.3831031, 0.4269175, -0.6323482, 0.7158800, -1.0989832, 1.0592657

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8437101, upper bound: 1.8335032
time: 1.63 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8437101, upper bound: 1.8335032
time: 1.94 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1_A1

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

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_B1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8841266, upper bound: 1.8443778
time: 1.72 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8850170, upper bound: 1.8480956
time: 2.29 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.2566187, 0.9745614, -0.7844378, 1.6371753, -1.8937941, 1.7589991
1: -0.3449113, 0.3714673, -0.7545584, 0.7572792, -1.1021905, 1.1260257
2: -0.4290012, 0.4467482, -0.8325227, 0.9933141, -1.4223152, 1.2792709
3: -0.3016051, 0.2813783, -0.6313741, 0.7623650, -1.0639701, 0.9127524
4: -0.3472668, 0.4047099, -0.7793881, 1.0365195, -1.3837863, 1.1840980
5: -0.4669960, 0.5325068, -0.9667646, 1.1034745, -1.5704705, 1.4992714
6: -0.1977043, 1.3230022, -1.0031563, 1.5844967, -1.7822011, 2.3261585
7: -0.3851508, 0.4853825, -0.9345270, 0.9043884, -1.2895392, 1.4199095
8: -0.3923964, 0.4462870, -0.8537888, 1.0604813, -1.4528776, 1.3000758
9: -0.2675011, 0.3124065, -0.7392236, 0.8510032, -1.1185043, 1.0516300

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_B1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8841266, upper bound: 1.8443778
time: 1.74 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8850170, upper bound: 1.8480955
time: 2.35 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.3736029, 1.2176384, -0.7594280, 1.6111423, -1.9847453, 1.9770663
1: -0.4629781, 0.4641406, -0.7370304, 0.7400607, -1.2030388, 1.2011710
2: -0.5556759, 0.5628081, -0.8156410, 0.9684181, -1.5240939, 1.3784492
3: -0.3796306, 0.3983214, -0.6158190, 0.7415797, -1.1212102, 1.0141404
4: -0.4801568, 0.4986325, -0.7614969, 1.0056032, -1.4857600, 1.2601295
5: -0.5810411, 0.6657445, -0.9424877, 1.0776376, -1.6586787, 1.6082323
6: -0.4606331, 1.4744508, -0.9701871, 1.5752594, -2.0358925, 2.4446378
7: -0.5150830, 0.6035222, -0.9103317, 0.8857892, -1.4008721, 1.5138539
8: -0.5282701, 0.5789487, -0.8326784, 1.0323420, -1.5606120, 1.4116272
9: -0.3740102, 0.4164491, -0.7191813, 0.8259785, -1.1999887, 1.1356304

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_B1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8845056, upper bound: 1.8478598
time: 2.06 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8845056, upper bound: 1.8478598
time: 2.63 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.3641120, 1.1754661, -0.7844378, 1.6371753, -2.0012875, 1.9599038
1: -0.4475572, 0.4595928, -0.7545584, 0.7572792, -1.2048364, 1.2141511
2: -0.5378050, 0.5554098, -0.8325227, 0.9933141, -1.5311191, 1.3879325
3: -0.3687481, 0.3912953, -0.6313741, 0.7623650, -1.1311131, 1.0226694
4: -0.4640032, 0.4938235, -0.7793881, 1.0365195, -1.5005227, 1.2732115
5: -0.5634810, 0.6611011, -0.9667646, 1.1034745, -1.6669555, 1.6278657
6: -0.4198382, 1.4364233, -1.0031563, 1.5844967, -2.0043349, 2.4395795
7: -0.5054985, 0.5889934, -0.9345270, 0.9043884, -1.4098868, 1.5235205
8: -0.5073896, 0.5716039, -0.8537888, 1.0604813, -1.5678709, 1.4253926
9: -0.3690580, 0.4140370, -0.7392236, 0.8510032, -1.2200612, 1.1532607

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_B1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8848065, upper bound: 1.8494081
time: 1.95 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8848065, upper bound: 1.8494081
time: 1.97 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3444705, 1.1611376, -0.5928441, 1.4439154, -1.7883859, 1.7539817
1: -0.4343950, 0.4384055, -0.6220707, 0.6239891, -1.0583841, 1.0604762
2: -0.5243341, 0.5331331, -0.7048268, 0.8009276, -1.3252617, 1.2379600
3: -0.3616669, 0.3646820, -0.5127134, 0.6018625, -0.9635293, 0.8773955
4: -0.4473096, 0.4675108, -0.6436394, 0.7953633, -1.2426729, 1.1111503
5: -0.5545209, 0.6285704, -0.7828684, 0.9058701, -1.4603910, 1.4114389
6: -0.3975932, 1.4438161, -0.7557898, 1.5220881, -1.9196813, 2.1996059
7: -0.4792118, 0.5759556, -0.7476259, 0.7641784, -1.2433902, 1.3235815
8: -0.4950513, 0.5437298, -0.6963511, 0.8437905, -1.3388418, 1.2400810
9: -0.3431973, 0.3873599, -0.5820532, 0.6570404, -1.0002377, 0.9694130

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_B2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8749391, upper bound: 1.8238898
time: 1.99 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8749391, upper bound: 1.8238898
time: 2.03 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3476995, 1.1663787, -0.7647165, 1.6365461, -1.9842457, 1.9310952
1: -0.4373175, 0.4413435, -0.7422393, 0.7379855, -1.1753030, 1.1835828
2: -0.5275090, 0.5365035, -0.8220410, 0.9675941, -1.4951031, 1.3585446
3: -0.3634359, 0.3685750, -0.6173563, 0.7416877, -1.1051235, 0.9859313
4: -0.4507725, 0.4704578, -0.7672406, 0.9997596, -1.4505321, 1.2376984
5: -0.5570605, 0.6328623, -0.9519932, 1.0745788, -1.6316392, 1.5848556
6: -0.4035295, 1.4458851, -0.9909197, 1.6031611, -2.0066905, 2.4368048
7: -0.4831508, 0.5787852, -0.9101856, 0.8915162, -1.3746670, 1.4889708
8: -0.4980760, 0.5477806, -0.8322191, 1.0311584, -1.5292344, 1.3799998
9: -0.3468383, 0.3908835, -0.7156761, 0.8218322, -1.1686704, 1.1065596

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_B2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8755558, upper bound: 1.8238689
time: 2.06 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8755558, upper bound: 1.8238689
time: 1.89 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.3781277, 1.2396262, -0.7672825, 1.6179278, -1.9960555, 2.0069087
1: -0.4709474, 0.4662821, -0.7425915, 0.7459769, -1.2169244, 1.2088735
2: -0.5652486, 0.5657305, -0.8206791, 0.9766487, -1.5418973, 1.3864095
3: -0.3850398, 0.4023046, -0.6218454, 0.7482450, -1.1332848, 1.0241499
4: -0.4889759, 0.5010123, -0.7666720, 1.0152607, -1.5042366, 1.2676843
5: -0.5902268, 0.6654201, -0.9496925, 1.0885332, -1.6787599, 1.6151125
6: -0.4818646, 1.4939808, -0.9801387, 1.5769963, -2.0588608, 2.4741194
7: -0.5199133, 0.6111791, -0.9179869, 0.8917904, -1.4117037, 1.5291660
8: -0.5396964, 0.5818094, -0.8414190, 1.0423281, -1.5820246, 1.4232285
9: -0.3778038, 0.4173222, -0.7246048, 0.8347075, -1.2125113, 1.1419270

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_B2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8752946, upper bound: 1.8296678
time: 4.08 seconds

## Relational analysis of IS_A1_B2_B2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8752946, upper bound: 1.8296678
time: 2.30 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.3376719, 1.1499140, -0.7672825, 1.6179278, -1.9555998, 1.9171965
1: -0.4281519, 0.4323016, -0.7425915, 0.7459769, -1.1741288, 1.1748930
2: -0.5175388, 0.5260578, -0.8206791, 0.9766487, -1.4941875, 1.3467369
3: -0.3578952, 0.3564347, -0.6218454, 0.7482450, -1.1061401, 0.9782801
4: -0.4398862, 0.4620605, -0.7666720, 1.0152607, -1.4551469, 1.2287326
5: -0.5491145, 0.6197780, -0.9496925, 1.0885332, -1.6376476, 1.5694705
6: -0.3851051, 1.4395692, -0.9801387, 1.5769963, -1.9621015, 2.4197078
7: -0.4709841, 0.5699319, -0.9179869, 0.8917904, -1.3627746, 1.4879187
8: -0.4887874, 0.5352499, -0.8414190, 1.0423281, -1.5311155, 1.3766689
9: -0.3354484, 0.3799347, -0.7246048, 0.8347075, -1.1701560, 1.1045396

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A1_B2_B2_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8752946, upper bound: 1.8309926
time: 2.11 seconds

## Relational analysis of IS_A1_B2_B2_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8752946, upper bound: 1.8309926
time: 2.20 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.3645582, 1.1859441, -0.7919115, 1.6434530, -2.0080113, 1.9778556
1: -0.4502758, 0.4584537, -0.7598349, 0.7628895, -1.2131653, 1.2182885
2: -0.5415984, 0.5544920, -0.8372648, 1.0011079, -1.5427064, 1.3917568
3: -0.3708249, 0.3911415, -0.6400046, 0.7686682, -1.1394931, 1.0311462
4: -0.4673262, 0.4921135, -0.7842556, 1.0456074, -1.5129337, 1.2763691
5: -0.5672812, 0.6562297, -0.9735982, 1.1139143, -1.6811955, 1.6298280
6: -0.4283218, 1.4462430, -1.0126145, 1.5860882, -2.0144100, 2.4588575
7: -0.5056106, 0.5915187, -0.9417365, 0.9100840, -1.4156946, 1.5332551
8: -0.5120641, 0.5695516, -0.8621641, 1.0699773, -1.5820414, 1.4317157
9: -0.3695081, 0.4113252, -0.7442783, 0.8592910, -1.2287991, 1.1556035

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8521304, upper bound: 1.8167881
time: 2.04 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8521304, upper bound: 1.8312210
time: 2.00 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.3275586, 1.1066502, -0.7919115, 1.6434530, -1.9710116, 1.8985617
1: -0.4122107, 0.4273778, -0.7598349, 0.7628895, -1.1751002, 1.1872127
2: -0.4990292, 0.5181047, -0.8372648, 1.0011079, -1.5001371, 1.3553694
3: -0.3466486, 0.3489987, -0.6400046, 0.7686682, -1.1153167, 0.9890033
4: -0.4231307, 0.4577282, -0.7842556, 1.0456074, -1.4687382, 1.2419838
5: -0.5308730, 0.6144143, -0.9735982, 1.1139143, -1.6447873, 1.5880125
6: -0.3435014, 1.4017423, -1.0126145, 1.5860882, -1.9295896, 2.4143567
7: -0.4610362, 0.5547385, -0.9417365, 0.9100840, -1.3711202, 1.4964750
8: -0.4676493, 0.5271855, -0.8621641, 1.0699773, -1.5376265, 1.3893497
9: -0.3302575, 0.3771160, -0.7442783, 0.8592910, -1.1895485, 1.1213944

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_B2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8340937, upper bound: 1.8210253
time: 2.21 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_B2_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8187598, upper bound: 1.7788462
time: 1.62 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.8010634, 1.6696776, -0.1992713, 0.9146871, -1.7157505, 1.8689489
1: -0.7670188, 0.7641536, -0.3016900, 0.3237556, -1.0907744, 1.0658436
2: -0.8456585, 1.0051625, -0.3902865, 0.3853621, -1.2310207, 1.3954489
3: -0.6399140, 0.7732075, -0.2729115, 0.2237414, -0.8636554, 1.0461190
4: -0.7929307, 1.0491253, -0.2951659, 0.3597881, -1.1527188, 1.3442912
5: -0.9862207, 1.1102405, -0.4355428, 0.4644218, -1.4506426, 1.5457833
6: -1.0347718, 1.6094160, -0.1188300, 1.3053600, -2.3401318, 1.7282460
7: -0.9469694, 0.9174546, -0.3232574, 0.4377548, -1.3847241, 1.2407119
8: -0.8610761, 1.0727658, -0.3480145, 0.3803610, -1.2414372, 1.4207804
9: -0.7488223, 0.8602611, -0.2095654, 0.2498562, -0.9986785, 1.0698265

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A2_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8199692, upper bound: 1.8392952
time: 2.01 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7805475, upper bound: 1.8239831
time: 1.50 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.8010634, 1.6696776, -0.2102968, 0.9211587, -1.7222221, 1.8799744
1: -0.7670188, 0.7641536, -0.3085759, 0.3338820, -1.1009008, 1.0727296
2: -0.8456585, 1.0051625, -0.3962048, 0.3978045, -1.2434629, 1.4013673
3: -0.6399140, 0.7732075, -0.2776511, 0.2353318, -0.8752458, 1.0508586
4: -0.7929307, 1.0491253, -0.3039144, 0.3694724, -1.1624031, 1.3530397
5: -0.9862207, 1.1102405, -0.4399044, 0.4803667, -1.4665874, 1.5501449
6: -1.0347718, 1.6094160, -0.1289688, 1.3063754, -2.3411472, 1.7383848
7: -0.9469694, 0.9174546, -0.3352553, 0.4456540, -1.3926234, 1.2527099
8: -0.8610761, 1.0727658, -0.3547675, 0.3940904, -1.2551665, 1.4275334
9: -0.7488223, 0.8602611, -0.2212234, 0.2636225, -1.0124449, 1.0814846

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A2_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8199692, upper bound: 1.8392952
time: 1.84 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7805475, upper bound: 1.8239831
time: 1.68 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6408266, 1.4699628, -0.3759446, 1.1946460, -1.8354726, 1.8459074
1: -0.6528313, 0.6626136, -0.4582976, 0.4706964, -1.1235278, 1.1209111
2: -0.7324177, 0.8555449, -0.5494688, 0.5677172, -1.3001349, 1.4050137
3: -0.5431786, 0.6473091, -0.3752226, 0.4059608, -0.9491394, 1.0225317
4: -0.6751659, 0.8704064, -0.4767937, 0.5077250, -1.1828909, 1.3472002
5: -0.8239776, 0.9598182, -0.5728263, 0.6767382, -1.5007159, 1.5326445
6: -0.8017728, 1.5047332, -0.4423463, 1.4436779, -2.2454507, 1.9470794
7: -0.8008890, 0.7960505, -0.5206935, 0.5994501, -1.4003391, 1.3167441
8: -0.7347701, 0.9041102, -0.5195796, 0.5863240, -1.3210940, 1.4236898
9: -0.6323482, 0.7158800, -0.3831031, 0.4269175, -1.0592657, 1.0989832

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8335032, upper bound: 1.8437101
time: 1.78 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8335032, upper bound: 1.8437101
time: 2.19 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_B1

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

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_A1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8443778, upper bound: 1.8841266
time: 1.87 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_B1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8480956, upper bound: 1.8850170
time: 1.74 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.7844378, 1.6371753, -0.2566187, 0.9745614, -1.7589991, 1.8937941
1: -0.7545584, 0.7572792, -0.3449113, 0.3714673, -1.1260257, 1.1021905
2: -0.8325227, 0.9933141, -0.4290012, 0.4467482, -1.2792709, 1.4223152
3: -0.6313741, 0.7623650, -0.3016051, 0.2813783, -0.9127524, 1.0639701
4: -0.7793881, 1.0365195, -0.3472668, 0.4047099, -1.1840980, 1.3837863
5: -0.9667646, 1.1034745, -0.4669960, 0.5325068, -1.4992714, 1.5704705
6: -1.0031563, 1.5844967, -0.1977043, 1.3230022, -2.3261585, 1.7822011
7: -0.9345270, 0.9043884, -0.3851508, 0.4853825, -1.4199095, 1.2895392
8: -0.8537888, 1.0604813, -0.3923964, 0.4462870, -1.3000758, 1.4528776
9: -0.7392236, 0.8510032, -0.2675011, 0.3124065, -1.0516300, 1.1185043

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_A1_B1_A2_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8443778, upper bound: 1.8841266
time: 2.12 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8480956, upper bound: 1.8850170
time: 2.05 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.7594280, 1.6111423, -0.3736029, 1.2176384, -1.9770663, 1.9847453
1: -0.7370304, 0.7400607, -0.4629781, 0.4641406, -1.2011710, 1.2030388
2: -0.8156410, 0.9684181, -0.5556759, 0.5628081, -1.3784492, 1.5240939
3: -0.6158190, 0.7415797, -0.3796306, 0.3983214, -1.0141404, 1.1212102
4: -0.7614969, 1.0056032, -0.4801568, 0.4986325, -1.2601295, 1.4857600
5: -0.9424877, 1.0776376, -0.5810411, 0.6657445, -1.6082323, 1.6586787
6: -0.9701871, 1.5752594, -0.4606331, 1.4744508, -2.4446378, 2.0358925
7: -0.9103317, 0.8857892, -0.5150830, 0.6035222, -1.5138539, 1.4008721
8: -0.8326784, 1.0323420, -0.5282701, 0.5789487, -1.4116272, 1.5606120
9: -0.7191813, 0.8259785, -0.3740102, 0.4164491, -1.1356304, 1.1999887

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A2_A1_B1_A2_B2_B1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8467583, upper bound: 1.8858963
time: 2.08 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_B1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8467583, upper bound: 1.8858963
time: 1.81 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.7844378, 1.6371753, -0.3641120, 1.1754661, -1.9599038, 2.0012875
1: -0.7545584, 0.7572792, -0.4475572, 0.4595928, -1.2141511, 1.2048364
2: -0.8325227, 0.9933141, -0.5378050, 0.5554098, -1.3879325, 1.5311191
3: -0.6313741, 0.7623650, -0.3687481, 0.3912953, -1.0226694, 1.1311131
4: -0.7793881, 1.0365195, -0.4640032, 0.4938235, -1.2732115, 1.5005227
5: -0.9667646, 1.1034745, -0.5634810, 0.6611011, -1.6278657, 1.6669555
6: -1.0031563, 1.5844967, -0.4198382, 1.4364233, -2.4395795, 2.0043349
7: -0.9345270, 0.9043884, -0.5054985, 0.5889934, -1.5235205, 1.4098868
8: -0.8537888, 1.0604813, -0.5073896, 0.5716039, -1.4253926, 1.5678709
9: -0.7392236, 0.8510032, -0.3690580, 0.4140370, -1.1532607, 1.2200612

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A2_A1_B1_A2_B2_B2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8488624, upper bound: 1.8861909
time: 2.38 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_B2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8488624, upper bound: 1.8861909
time: 1.90 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.53 seconds
IS_A1_B1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8882222, upper bound: 1.8621048
IS_A1_B1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8618688, upper bound: 1.8617999
IS_A1_B1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8899332, upper bound: 1.8627186
IS_A1_B1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8622863, upper bound: 1.8624052
IS_A1_B1_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8843058, upper bound: 1.8495865
IS_A1_B1_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8609583, upper bound: 1.8492783
IS_A1_B1_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8884776, upper bound: 1.8541485
IS_A1_B1_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8620255, upper bound: 1.8538202
IS_A1_B1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8628802, upper bound: 1.8899332
IS_A1_B1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8626091, upper bound: 1.8623924
IS_A1_B1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8632006, upper bound: 1.8899332
IS_A1_B1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8629862, upper bound: 1.8629863
IS_A1_B1_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8622780, upper bound: 1.8802745
IS_A1_B1_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8618168, upper bound: 1.8499085
IS_A1_B1_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8884776, upper bound: 1.8546595
IS_A1_B1_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8627467, upper bound: 1.8544514
IS_A1_B1_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8528223, upper bound: 1.8851599
IS_A1_B1_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8525192, upper bound: 1.8610949
IS_A1_B1_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8810647, upper bound: 1.8623144
IS_A1_B1_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8531095, upper bound: 1.8618521
IS_A1_B1_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8526566, upper bound: 1.8772503
IS_A1_B1_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8522712, upper bound: 1.8491903
IS_A1_B1_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8809846, upper bound: 1.8540616
IS_A1_B1_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8530874, upper bound: 1.8537294
IS_A1_B1_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8802745, upper bound: 1.8622780
IS_A1_B1_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8499085, upper bound: 1.8618168
IS_A1_B1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8546595, upper bound: 1.8884776
IS_A1_B1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8544515, upper bound: 1.8627467
IS_A1_B1_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8801938, upper bound: 1.8540221
IS_A1_B1_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8812708, upper bound: 1.8546285
IS_A1_B1_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8498948, upper bound: 1.8536285
IS_A1_B1_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8544282, upper bound: 1.8544297
IS_A1_B2_B1_B1_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8392952, upper bound: 1.8199692
IS_A1_B2_B1_B1_A1_A1_B2, status: Status.VERIFIED, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8239831, upper bound: 1.7805475
IS_A1_B2_B1_B1_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8392952, upper bound: 1.8199692
IS_A1_B2_B1_B1_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8239831, upper bound: 1.7805475
IS_A1_B2_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8437101, upper bound: 1.8335032
IS_A1_B2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8437101, upper bound: 1.8335032
IS_A1_B2_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8841266, upper bound: 1.8443778
IS_A1_B2_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8850170, upper bound: 1.8480956
IS_A1_B2_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8841266, upper bound: 1.8443778
IS_A1_B2_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8850170, upper bound: 1.8480955
IS_A1_B2_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8845056, upper bound: 1.8478598
IS_A1_B2_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8845056, upper bound: 1.8478598
IS_A1_B2_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8848065, upper bound: 1.8494081
IS_A1_B2_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8848065, upper bound: 1.8494081
IS_A1_B2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8749391, upper bound: 1.8238898
IS_A1_B2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8749391, upper bound: 1.8238898
IS_A1_B2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8755558, upper bound: 1.8238689
IS_A1_B2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8755558, upper bound: 1.8238689
IS_A1_B2_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8752946, upper bound: 1.8296678
IS_A1_B2_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8752946, upper bound: 1.8296678
IS_A1_B2_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8752946, upper bound: 1.8309926
IS_A1_B2_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8752946, upper bound: 1.8309926
IS_A1_B2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8521304, upper bound: 1.8167881
IS_A1_B2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8521304, upper bound: 1.8312210
IS_A1_B2_B2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8340937, upper bound: 1.8210253
IS_A1_B2_B2_B2_A2_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8187598, upper bound: 1.7788462
IS_A2_A1_B1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8199692, upper bound: 1.8392952
IS_A2_A1_B1_A1_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.7805475, upper bound: 1.8239831
IS_A2_A1_B1_A1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8199692, upper bound: 1.8392952
IS_A2_A1_B1_A1_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.7805475, upper bound: 1.8239831
IS_A2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8335032, upper bound: 1.8437101
IS_A2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8335032, upper bound: 1.8437101
IS_A2_A1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8443778, upper bound: 1.8841266
IS_A2_A1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8480956, upper bound: 1.8850170
IS_A2_A1_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8443778, upper bound: 1.8841266
IS_A2_A1_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8480956, upper bound: 1.8850170
IS_A2_A1_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8467583, upper bound: 1.8858963
IS_A2_A1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8467583, upper bound: 1.8858963
IS_A2_A1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8488624, upper bound: 1.8861909
IS_A2_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 6, lower bound: -1.8488624, upper bound: 1.8861909

## BFS IS instance: IS_A1_B1_A1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1154602, 0.7481776, -0.2240654, 0.9332255, -1.0486857, 0.9722430
1: -0.2176643, 0.2517601, -0.3184277, 0.3449340, -0.5625983, 0.5701878
2: -0.3075277, 0.2953086, -0.4045457, 0.4120911, -0.7196187, 0.6998543
3: -0.2176826, 0.1301930, -0.2840719, 0.2497660, -0.4674486, 0.4142649
4: -0.2050004, 0.2960963, -0.3166633, 0.3790966, -0.5840970, 0.6127596
5: -0.3598376, 0.3728185, -0.4459834, 0.4924448, -0.8522824, 0.8188019
6: 0.0682609, 1.2514322, -0.1463477, 1.3040482, -1.2357873, 1.3977799
7: -0.2254871, 0.3546931, -0.3504135, 0.4569526, -0.6824396, 0.7051066
8: -0.2653745, 0.2764327, -0.3645772, 0.4085877, -0.6739621, 0.6410099
9: -0.1340322, 0.1886803, -0.2373872, 0.2784784, -0.4125106, 0.4260675

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8751320, upper bound: 1.8425451
time: 5.07 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8785460, upper bound: 1.8531280
time: 2.05 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1142295, 0.6751896, -0.1889416, 0.8809597, -0.9951892, 0.8641311
1: -0.1899875, 0.2309581, -0.2879402, 0.3177280, -0.5077155, 0.5188984
2: -0.2761208, 0.2664815, -0.3753113, 0.3762129, -0.6523337, 0.6417928
3: -0.1968510, 0.1122572, -0.2634344, 0.2166701, -0.4135211, 0.3756915
4: -0.1744129, 0.2778596, -0.2814730, 0.3532992, -0.5277121, 0.5593327
5: -0.3298081, 0.3514048, -0.4202585, 0.4542189, -0.7840270, 0.7716633
6: 0.1416302, 1.2488258, -0.0843956, 1.2809023, -1.1392720, 1.3332214
7: -0.1968541, 0.3268532, -0.3135076, 0.4243550, -0.6212091, 0.6403608
8: -0.2415736, 0.2475373, -0.3311545, 0.3700297, -0.6116033, 0.5786918
9: -0.1150313, 0.1769733, -0.2055897, 0.2449263, -0.3599576, 0.3825630

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8517709, upper bound: 1.8422138
time: 1.75 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8528454, upper bound: 1.8528165
time: 1.83 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1227664, 0.7666053, -0.1910335, 0.8744125, -0.9971789, 0.9576389
1: -0.2257826, 0.2619199, -0.2874907, 0.3210520, -0.5468345, 0.5494106
2: -0.3161956, 0.3064210, -0.3742104, 0.3797177, -0.6959133, 0.6806314
3: -0.2234198, 0.1415143, -0.2629691, 0.2196814, -0.4431013, 0.4044835
4: -0.2142171, 0.3038808, -0.2811005, 0.3570620, -0.5712792, 0.5849812
5: -0.3674277, 0.3825197, -0.4179860, 0.4617836, -0.8292112, 0.8005056
6: 0.0477321, 1.2551593, -0.0810578, 1.2809436, -1.2332115, 1.3362170
7: -0.2376474, 0.3620259, -0.3162356, 0.4242360, -0.6618834, 0.6782615
8: -0.2744759, 0.2878318, -0.3359746, 0.3749803, -0.6494563, 0.6238064
9: -0.1429872, 0.1947979, -0.2082598, 0.2505935, -0.3935808, 0.4030577

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8882222, upper bound: 1.8623181
time: 1.94 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8882222, upper bound: 1.8627186
time: 1.99 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1155493, 0.6945989, -0.1552477, 0.8210121, -0.9365614, 0.8498466
1: -0.1972181, 0.2371659, -0.2562303, 0.2925962, -0.4898142, 0.4933962
2: -0.2841565, 0.2776476, -0.3445398, 0.3426173, -0.6267738, 0.6221874
3: -0.2030154, 0.1174017, -0.2418194, 0.1849302, -0.3879457, 0.3592210
4: -0.1830648, 0.2839549, -0.2449292, 0.3312224, -0.5142872, 0.5288842
5: -0.3367375, 0.3602239, -0.3919571, 0.4235481, -0.7602856, 0.7521811
6: 0.1221733, 1.2521865, -0.0172234, 1.2698956, -1.1477222, 1.2694099
7: -0.2062643, 0.3342494, -0.2786800, 0.3901888, -0.5964531, 0.6129294
8: -0.2500136, 0.2584260, -0.3050335, 0.3342662, -0.5842798, 0.5634595
9: -0.1224575, 0.1823628, -0.1751183, 0.2183368, -0.3407943, 0.3574811

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8609757, upper bound: 1.8619508
time: 2.26 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8609757, upper bound: 1.8624052
time: 2.40 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1154602, 0.7481776, -0.2430249, 0.9576500, -1.0731102, 0.9912026
1: -0.2176643, 0.2517601, -0.3340049, 0.3605127, -0.5781770, 0.5857650
2: -0.3075277, 0.2953086, -0.4189556, 0.4324076, -0.7399352, 0.7142642
3: -0.2176826, 0.1301930, -0.2943027, 0.2680599, -0.4857425, 0.4244957
4: -0.2050004, 0.2960963, -0.3343638, 0.3944111, -0.5994115, 0.6304600
5: -0.3598376, 0.3728185, -0.4582541, 0.5166931, -0.8765308, 0.8310726
6: 0.0682609, 1.2514322, -0.1768547, 1.3163049, -1.2480440, 1.4282868
7: -0.2254871, 0.3546931, -0.3706639, 0.4735220, -0.6990091, 0.7253569
8: -0.2653745, 0.2764327, -0.3809146, 0.4310794, -0.6964538, 0.6573472
9: -0.1340322, 0.1886803, -0.2544016, 0.2984248, -0.4324570, 0.4430819

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8696515, upper bound: 1.8322504
time: 2.36 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8740476, upper bound: 1.8407304
time: 1.85 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1142295, 0.6751896, -0.2083534, 0.9052123, -1.0194418, 0.8835430
1: -0.1899875, 0.2309581, -0.3039190, 0.3336554, -0.5236429, 0.5348771
2: -0.2761208, 0.2664815, -0.3901527, 0.3969911, -0.6731120, 0.6566342
3: -0.1968510, 0.1122572, -0.2739803, 0.2354785, -0.4323295, 0.3862374
4: -0.1744129, 0.2778596, -0.2995957, 0.3689795, -0.5433924, 0.5774554
5: -0.3298081, 0.3514048, -0.4326246, 0.4790466, -0.8088547, 0.7840294
6: 0.1416302, 1.2488258, -0.1158310, 1.2926724, -1.1510422, 1.3646568
7: -0.1968541, 0.3268532, -0.3343637, 0.4412498, -0.6381040, 0.6612169
8: -0.2415736, 0.2475373, -0.3502313, 0.3930596, -0.6346331, 0.5977686
9: -0.1150313, 0.1769733, -0.2230937, 0.2654258, -0.3804571, 0.4000670

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8499831, upper bound: 1.8319271
time: 1.54 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8519780, upper bound: 1.8404128
time: 1.81 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1227664, 0.7666053, -0.1928261, 0.8672711, -0.9900375, 0.9594314
1: -0.2257826, 0.2619199, -0.2870591, 0.3239554, -0.5497380, 0.5489790
2: -0.3161956, 0.3064210, -0.3726139, 0.3829908, -0.6991864, 0.6790349
3: -0.2234198, 0.1415143, -0.2624988, 0.2228808, -0.4463006, 0.4040132
4: -0.2142171, 0.3038808, -0.2809156, 0.3601927, -0.5744098, 0.5847963
5: -0.3674277, 0.3825197, -0.4151758, 0.4674203, -0.8348480, 0.7976955
6: 0.0477321, 1.2551593, -0.0771782, 1.2824421, -1.2347100, 1.3323375
7: -0.2376474, 0.3620259, -0.3191629, 0.4238093, -0.6614567, 0.6811888
8: -0.2744759, 0.2878318, -0.3392406, 0.3791988, -0.6536747, 0.6270724
9: -0.1429872, 0.1947979, -0.2118111, 0.2561131, -0.3991004, 0.4066090

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8843058, upper bound: 1.8535278
time: 2.01 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8843058, upper bound: 1.8541485
time: 2.49 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1155493, 0.6945989, -0.1569195, 0.8138011, -0.9293504, 0.8515184
1: -0.1972181, 0.2371659, -0.2557466, 0.2959449, -0.4931629, 0.4929125
2: -0.2841565, 0.2776476, -0.3428630, 0.3461748, -0.6303313, 0.6205107
3: -0.2030154, 0.1174017, -0.2413020, 0.1888038, -0.3918193, 0.3587037
4: -0.1830648, 0.2839549, -0.2447421, 0.3339843, -0.5170491, 0.5286970
5: -0.3367375, 0.3602239, -0.3891134, 0.4285682, -0.7653057, 0.7493373
6: 0.1221733, 1.2521865, -0.0139928, 1.2745811, -1.1524078, 1.2661793
7: -0.2062643, 0.3342494, -0.2814012, 0.3901917, -0.5964559, 0.6156507
8: -0.2500136, 0.2584260, -0.3080465, 0.3392050, -0.5892186, 0.5664725
9: -0.1224575, 0.1823628, -0.1792819, 0.2222409, -0.3446985, 0.3616448

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8601397, upper bound: 1.8530939
time: 1.95 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8601397, upper bound: 1.8538200
time: 2.11 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.1908983, 0.8756229, -0.1354999, 0.7824346, -0.9733329, 1.0111228
1: -0.2876973, 0.3208417, -0.2365710, 0.2770546, -0.5647519, 0.5574127
2: -0.3745519, 0.3795155, -0.3247362, 0.3232960, -0.6978478, 0.7042516
3: -0.2631456, 0.2193505, -0.2294425, 0.1641774, -0.4273230, 0.4487929
4: -0.2811765, 0.3569371, -0.2258819, 0.3156322, -0.5968087, 0.5828190
5: -0.4184634, 0.4617238, -0.3733644, 0.3993013, -0.8177647, 0.8350882
6: -0.0819756, 1.2819579, 0.0263824, 1.2539568, -1.3359324, 1.2555754
7: -0.3160120, 0.4243731, -0.2580026, 0.3711661, -0.6871781, 0.6823757
8: -0.3360884, 0.3748245, -0.2854451, 0.3072547, -0.6433431, 0.6602696
9: -0.2076989, 0.2501320, -0.1600177, 0.2056232, -0.4133221, 0.4101497

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8623181, upper bound: 1.8899332
time: 2.08 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8623181, upper bound: 1.8899332
time: 5.31 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.1553052, 0.8222291, -0.1163760, 0.7126382, -0.8679434, 0.9386051
1: -0.2565348, 0.2925226, -0.2053943, 0.2552311, -0.5117659, 0.4979169
2: -0.3449629, 0.3426450, -0.2935745, 0.2953720, -0.6403350, 0.6362195
3: -0.2421206, 0.1845892, -0.2094344, 0.1342419, -0.3763625, 0.3940236
4: -0.2452519, 0.3312366, -0.1940669, 0.2962217, -0.5414736, 0.5253035
5: -0.3924647, 0.4237420, -0.3432568, 0.3717371, -0.7642018, 0.7669988
6: -0.0182145, 1.2706414, 0.0990996, 1.2520018, -1.2702162, 1.1715419
7: -0.2786186, 0.3905089, -0.2257933, 0.3416140, -0.6202326, 0.6163023
8: -0.3053528, 0.3340982, -0.2615497, 0.2755364, -0.5808892, 0.5956479
9: -0.1746943, 0.2183504, -0.1396160, 0.1913607, -0.3660550, 0.3579663

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8619508, upper bound: 1.8612699
time: 1.57 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8619508, upper bound: 1.8623924
time: 1.74 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.2072586, 0.8959078, -0.1182266, 0.7250099, -0.9322685, 1.0141344
1: -0.3009212, 0.3340663, -0.2100975, 0.2579224, -0.5588436, 0.5441638
2: -0.3868112, 0.3966483, -0.2988325, 0.2990227, -0.6858339, 0.6954808
3: -0.2720728, 0.2352810, -0.2129812, 0.1366865, -0.4087593, 0.4482622
4: -0.2967181, 0.3695163, -0.1985588, 0.2995346, -0.5962527, 0.5680751
5: -0.4288297, 0.4805791, -0.3484361, 0.3773563, -0.8061860, 0.8290153
6: -0.1073784, 1.2879556, 0.0871900, 1.2580097, -1.3653882, 1.2007656
7: -0.3334463, 0.4387536, -0.2293885, 0.3463866, -0.6798328, 0.6681421
8: -0.3504367, 0.3934267, -0.2667631, 0.2803227, -0.6307594, 0.6601898
9: -0.2233384, 0.2670660, -0.1407366, 0.1940934, -0.4174318, 0.4078027

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8627186, upper bound: 1.8899332
time: 2.74 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8627186, upper bound: 1.8899332
time: 1.88 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.1701777, 0.8417210, -0.1199900, 0.6443160, -0.8144937, 0.9617110
1: -0.2688829, 0.3054476, -0.1785731, 0.2290163, -0.4978992, 0.4840207
2: -0.3562117, 0.3590200, -0.2636494, 0.2652339, -0.6214457, 0.6226695
3: -0.2504892, 0.2004787, -0.1923432, 0.1082504, -0.3587397, 0.3928220
4: -0.2598514, 0.3422543, -0.1638731, 0.2764139, -0.5362654, 0.5061274
5: -0.4021902, 0.4402497, -0.3148347, 0.3501648, -0.7523550, 0.7550845
6: -0.0428542, 1.2748969, 0.1694297, 1.2598227, -1.3026769, 1.1054672
7: -0.2946207, 0.4044873, -0.1926194, 0.3156230, -0.6102437, 0.5971068
8: -0.3187363, 0.3527098, -0.2378494, 0.2463247, -0.5650610, 0.5905592
9: -0.1900038, 0.2317110, -0.1165799, 0.1793276, -0.3693314, 0.3482910

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8624052, upper bound: 1.8622863
time: 2.15 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8624052, upper bound: 1.8629863
time: 2.18 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.1908983, 0.8756229, -0.1496533, 0.8027170, -0.9936153, 1.0252762
1: -0.2876973, 0.3208417, -0.2492483, 0.2891380, -0.5768353, 0.5700900
2: -0.3745519, 0.3795155, -0.3365934, 0.3378147, -0.7123666, 0.7161089
3: -0.2631456, 0.2193505, -0.2367675, 0.1814824, -0.4446279, 0.4561179
4: -0.2811765, 0.3569371, -0.2378066, 0.3275062, -0.6086827, 0.5947437
5: -0.4184634, 0.4617238, -0.3834556, 0.4174277, -0.8358911, 0.8451794
6: -0.0819756, 1.2819579, 0.0000213, 1.2638035, -1.3457791, 1.2819365
7: -0.3160120, 0.4243731, -0.2738677, 0.3830705, -0.6990824, 0.6982408
8: -0.3360884, 0.3748245, -0.2990615, 0.3285752, -0.6646636, 0.6738859
9: -0.2076989, 0.2501320, -0.1734349, 0.2157197, -0.4234187, 0.4235669

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8613575, upper bound: 1.8796464
time: 1.78 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8613575, upper bound: 1.8802744
time: 2.03 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.1553052, 0.8222291, -0.1211378, 0.7363443, -0.8916495, 0.9433670
1: -0.2565348, 0.2925226, -0.2184544, 0.2689328, -0.5254676, 0.5109770
2: -0.3449629, 0.3426450, -0.3056295, 0.3116248, -0.6565877, 0.6482744
3: -0.2421206, 0.1845892, -0.2179344, 0.1521668, -0.3942874, 0.4025236
4: -0.2452519, 0.3312366, -0.2075353, 0.3085364, -0.5537883, 0.5387719
5: -0.3924647, 0.4237420, -0.3532402, 0.3911815, -0.7836462, 0.7769822
6: -0.0182145, 1.2706414, 0.0701816, 1.2632302, -1.2814448, 1.2004598
7: -0.2786186, 0.3905089, -0.2441529, 0.3532750, -0.6318936, 0.6346618
8: -0.3053528, 0.3340982, -0.2762461, 0.2940101, -0.5993629, 0.6103444
9: -0.1746943, 0.2183504, -0.1515249, 0.2029680, -0.3776623, 0.3698753

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8598876, upper bound: 1.8474190
time: 1.79 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8598876, upper bound: 1.8499085
time: 1.75 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1185406, 0.7293256, -0.2067769, 0.8844675, -1.0030081, 0.9361025
1: -0.2119833, 0.2593544, -0.2982824, 0.3353040, -0.5472872, 0.5576369
2: -0.3007635, 0.3008489, -0.3830329, 0.3976636, -0.6984271, 0.6838818
3: -0.2142388, 0.1384656, -0.2701186, 0.2365419, -0.4507807, 0.4085842
4: -0.2005322, 0.3008204, -0.2941514, 0.3709590, -0.5714912, 0.5949717
5: -0.3502814, 0.3793796, -0.4239522, 0.4835648, -0.8338462, 0.8033319
6: 0.0826469, 1.2590265, -0.0985513, 1.2864097, -1.2037628, 1.3575778
7: -0.2314057, 0.3481143, -0.3340706, 0.4360552, -0.6674610, 0.6821849
8: -0.2685392, 0.2823639, -0.3514939, 0.3950947, -0.6636338, 0.6338578
9: -0.1418016, 0.1952014, -0.2252686, 0.2706831, -0.4124846, 0.4204700

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8843420, upper bound: 1.8541970
time: 1.94 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8843420, upper bound: 1.8546595
time: 2.06 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1202125, 0.6485925, -0.1703098, 0.8303915, -0.9506040, 0.8189023
1: -0.1802022, 0.2302629, -0.2667398, 0.3071526, -0.4873547, 0.4970027
2: -0.2654942, 0.2669307, -0.3528548, 0.3606301, -0.6261243, 0.6197854
3: -0.1933940, 0.1093308, -0.2488675, 0.2023865, -0.3957806, 0.3581983
4: -0.1656640, 0.2775859, -0.2578618, 0.3441591, -0.5098231, 0.5354476
5: -0.3165918, 0.3518349, -0.3974828, 0.4438551, -0.7604469, 0.7493177
6: 0.1652406, 1.2603061, -0.0350866, 1.2779989, -1.1127584, 1.2953928
7: -0.1942368, 0.3173171, -0.2959270, 0.4023034, -0.5965402, 0.6132442
8: -0.2395092, 0.2481518, -0.3202327, 0.3550710, -0.5945803, 0.5683845
9: -0.1175643, 0.1801665, -0.1925730, 0.2360458, -0.3536101, 0.3727396

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8612797, upper bound: 1.8538901
time: 2.16 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8612797, upper bound: 1.8496140
time: 3.23 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.2104456, 0.9213287, -0.1257720, 0.7672151, -0.9776608, 1.0471007
1: -0.3086960, 0.3340848, -0.2275481, 0.2682222, -0.5769182, 0.5616329
2: -0.3963239, 0.3979612, -0.3165084, 0.3127923, -0.7091162, 0.7144696
3: -0.2777308, 0.2354796, -0.2241413, 0.1514692, -0.4292001, 0.4596209
4: -0.3040550, 0.3697326, -0.2169592, 0.3077132, -0.6117682, 0.5866918
5: -0.4399903, 0.4805626, -0.3664474, 0.3875120, -0.8275023, 0.8470100
6: -0.1291885, 1.3064616, 0.0453056, 1.2506304, -1.3798189, 1.2611561
7: -0.3354445, 0.4457849, -0.2461038, 0.3626115, -0.6980560, 0.6918887
8: -0.3548898, 0.3944669, -0.2767595, 0.2939097, -0.6487995, 0.6712264
9: -0.2213693, 0.2639343, -0.1504686, 0.1990541, -0.4204234, 0.4144029

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8511876, upper bound: 1.8775614
time: 1.72 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8528223, upper bound: 1.8851599
time: 2.07 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.1749096, 0.8690729, -0.1150155, 0.6977203, -0.8726299, 0.9840884
1: -0.2778723, 0.3066671, -0.1981744, 0.2467533, -0.5246256, 0.5048416
2: -0.3668301, 0.3618361, -0.2864854, 0.2856790, -0.6525090, 0.6483215
3: -0.2568948, 0.2020085, -0.2046618, 0.1234181, -0.3803129, 0.4066704
4: -0.2686115, 0.3435809, -0.1858859, 0.2897382, -0.5583497, 0.5294668
5: -0.4145326, 0.4418157, -0.3371114, 0.3627391, -0.7772717, 0.7789271
6: -0.0667380, 1.2848203, 0.1164206, 1.2490261, -1.3157641, 1.1683997
7: -0.2980355, 0.4130357, -0.2148101, 0.3356073, -0.6336428, 0.6278458
8: -0.3223648, 0.3553537, -0.2539575, 0.2653587, -0.5877235, 0.6093111
9: -0.1893111, 0.2298907, -0.1320381, 0.1855236, -0.3748348, 0.3619288

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8491535, upper bound: 1.8479453
time: 1.79 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8525192, upper bound: 1.8610949
time: 2.13 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1269446, 0.7707946, -0.1910335, 0.8744125, -1.0013571, 0.9618282
1: -0.2289524, 0.2688800, -0.2874907, 0.3210520, -0.5500043, 0.5563706
2: -0.3186322, 0.3139849, -0.3742104, 0.3797177, -0.6983500, 0.6881953
3: -0.2254873, 0.1506825, -0.2629691, 0.2196814, -0.4451688, 0.4136516
4: -0.2181060, 0.3095212, -0.2811005, 0.3570620, -0.5751680, 0.5906217
5: -0.3687969, 0.3922480, -0.4179860, 0.4617836, -0.8305805, 0.8102340
6: 0.0414976, 1.2613519, -0.0810578, 1.2809436, -1.2394459, 1.3424097
7: -0.2462607, 0.3639614, -0.3162356, 0.4242360, -0.6704967, 0.6801970
8: -0.2809784, 0.2963188, -0.3359746, 0.3749803, -0.6559587, 0.6322935
9: -0.1491844, 0.2012617, -0.2082598, 0.2505935, -0.3997779, 0.4095215

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8781258, upper bound: 1.8613575
time: 1.81 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8781258, upper bound: 1.8623144
time: 1.83 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1193608, 0.6982825, -0.1552477, 0.8210121, -0.9403728, 0.8535302
1: -0.1990476, 0.2447662, -0.2562303, 0.2925962, -0.4916438, 0.5009965
2: -0.2869840, 0.2844429, -0.3445398, 0.3426173, -0.6296013, 0.6289827
3: -0.2053713, 0.1220908, -0.2418194, 0.1849302, -0.3903015, 0.3639101
4: -0.1858881, 0.2900116, -0.2449292, 0.3312224, -0.5171105, 0.5349408
5: -0.3381466, 0.3670229, -0.3919571, 0.4235481, -0.7616947, 0.7589800
6: 0.1163301, 1.2606424, -0.0172234, 1.2698956, -1.1535654, 1.2778659
7: -0.2129659, 0.3362022, -0.2786800, 0.3901888, -0.6031547, 0.6148822
8: -0.2562447, 0.2660408, -0.3050335, 0.3342662, -0.5905110, 0.5710742
9: -0.1285460, 0.1873699, -0.1751183, 0.2183368, -0.3468828, 0.3624882

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8471214, upper bound: 1.8598876
time: 1.98 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8471214, upper bound: 1.8618521
time: 1.85 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.2104456, 0.9213287, -0.1392433, 0.7876062, -0.9980519, 1.0605720
1: -0.3086960, 0.3340848, -0.2400220, 0.2804212, -0.5891172, 0.5741068
2: -0.3963239, 0.3979612, -0.3280258, 0.3275103, -0.7238342, 0.7259870
3: -0.2777308, 0.2354796, -0.2315183, 0.1682647, -0.4459955, 0.4669979
4: -0.3040550, 0.3697326, -0.2289880, 0.3193088, -0.6233637, 0.5987206
5: -0.4399903, 0.4805626, -0.3759934, 0.4058381, -0.8458283, 0.8565560
6: -0.1291885, 1.3064616, 0.0188218, 1.2604506, -1.3896391, 1.2876399
7: -0.3354445, 0.4457849, -0.2621655, 0.3743486, -0.7097931, 0.7079504
8: -0.3548898, 0.3944669, -0.2905442, 0.3131096, -0.6679994, 0.6850110
9: -0.2213693, 0.2639343, -0.1627370, 0.2091895, -0.4305588, 0.4266713

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8494285, upper bound: 1.8635489
time: 2.07 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8526566, upper bound: 1.8772503
time: 1.90 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.1749096, 0.8690729, -0.1195543, 0.7215676, -0.8964772, 0.9886272
1: -0.2778723, 0.3066671, -0.2100253, 0.2604244, -0.5382967, 0.5166924
2: -0.3668301, 0.3618361, -0.2980938, 0.3015589, -0.6683890, 0.6599299
3: -0.2568948, 0.2020085, -0.2127254, 0.1406005, -0.3974954, 0.4147340
4: -0.2686115, 0.3435809, -0.1989048, 0.3011509, -0.5697624, 0.5424858
5: -0.4145326, 0.4418157, -0.3471000, 0.3798417, -0.7943743, 0.7889158
6: -0.0667380, 1.2848203, 0.0880864, 1.2600918, -1.3268298, 1.1967340
7: -0.2980355, 0.4130357, -0.2326634, 0.3452882, -0.6433237, 0.6456991
8: -0.3223648, 0.3553537, -0.2678868, 0.2830926, -0.6054574, 0.6232405
9: -0.1893111, 0.2298907, -0.1434938, 0.1965707, -0.3858818, 0.3733845

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8339739, upper bound: 1.8391627
time: 1.87 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8437225, upper bound: 1.8403100
time: 1.90 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1269446, 0.7707946, -0.1928261, 0.8672711, -0.9942157, 0.9636208
1: -0.2289524, 0.2688800, -0.2870591, 0.3239554, -0.5529078, 0.5559390
2: -0.3186322, 0.3139849, -0.3726139, 0.3829908, -0.7016230, 0.6865988
3: -0.2254873, 0.1506825, -0.2624988, 0.2228808, -0.4483681, 0.4131812
4: -0.2181060, 0.3095212, -0.2809156, 0.3601927, -0.5782987, 0.5904368
5: -0.3687969, 0.3922480, -0.4151758, 0.4674203, -0.8362172, 0.8074238
6: 0.0414976, 1.2613519, -0.0771782, 1.2824421, -1.2409444, 1.3385302
7: -0.2462607, 0.3639614, -0.3191629, 0.4238093, -0.6700700, 0.6831243
8: -0.2809784, 0.2963188, -0.3392406, 0.3791988, -0.6601772, 0.6355594
9: -0.1491844, 0.2012617, -0.2118111, 0.2561131, -0.4052975, 0.4130728

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8712049, upper bound: 1.8525946
time: 2.36 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8809846, upper bound: 1.8540616
time: 1.96 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1193608, 0.6982825, -0.1569195, 0.8138011, -0.9331619, 0.8552020
1: -0.1990476, 0.2447662, -0.2557466, 0.2959449, -0.4949925, 0.5005128
2: -0.2869840, 0.2844429, -0.3428630, 0.3461748, -0.6331587, 0.6273060
3: -0.2053713, 0.1220908, -0.2413020, 0.1888038, -0.3941751, 0.3633928
4: -0.1858881, 0.2900116, -0.2447421, 0.3339843, -0.5198724, 0.5347537
5: -0.3381466, 0.3670229, -0.3891134, 0.4285682, -0.7667148, 0.7561363
6: 0.1163301, 1.2606424, -0.0139928, 1.2745811, -1.1582509, 1.2746353
7: -0.2129659, 0.3362022, -0.2814012, 0.3901917, -0.6031576, 0.6176034
8: -0.2562447, 0.2660408, -0.3080465, 0.3392050, -0.5954497, 0.5740873
9: -0.1285460, 0.1873699, -0.1792819, 0.2222409, -0.3507870, 0.3666518

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8469727, upper bound: 1.8526206
time: 1.99 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8469727, upper bound: 1.8537294
time: 1.84 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.1313374, 0.7761559, -0.2172361, 0.9069000, -1.0382375, 0.9933920
1: -0.2329237, 0.2736253, -0.3086752, 0.3421522, -0.5750759, 0.5823004
2: -0.3215864, 0.3194979, -0.3938075, 0.4071273, -0.7287136, 0.7133054
3: -0.2275660, 0.1578354, -0.2772135, 0.2452471, -0.4728131, 0.4350489
4: -0.2221345, 0.3132132, -0.3061112, 0.3769857, -0.5991203, 0.6193244
5: -0.3706363, 0.3974245, -0.4345517, 0.4914089, -0.8620452, 0.8319761
6: 0.0332642, 1.2595178, -0.1216769, 1.2899801, -1.2567158, 1.3811946
7: -0.2528917, 0.3676515, -0.3441595, 0.4473241, -0.7002158, 0.7118110
8: -0.2844544, 0.3018163, -0.3585033, 0.4045241, -0.6889785, 0.6603196
9: -0.1543849, 0.2044106, -0.2335665, 0.2775798, -0.4319648, 0.4379771

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8776334, upper bound: 1.8614700
time: 1.99 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8776334, upper bound: 1.8622780
time: 2.68 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.1192040, 0.7101219, -0.1801198, 0.8523152, -0.9715191, 0.8902417
1: -0.2041196, 0.2537934, -0.2766441, 0.3134366, -0.5175562, 0.5304376
2: -0.2926871, 0.2937365, -0.3632229, 0.3693452, -0.6620324, 0.6569594
3: -0.2091788, 0.1312812, -0.2556435, 0.2103998, -0.4195786, 0.3869247
4: -0.1920935, 0.2962902, -0.2691517, 0.3497238, -0.5418173, 0.5654420
5: -0.3424911, 0.3728835, -0.4075813, 0.4510895, -0.7935807, 0.7804649
6: 0.1018389, 1.2595706, -0.0571852, 1.2756747, -1.1738358, 1.3167558
7: -0.2233809, 0.3407225, -0.3053441, 0.4129571, -0.6363381, 0.6460667
8: -0.2625234, 0.2748557, -0.3267936, 0.3638026, -0.6263260, 0.6016493
9: -0.1373341, 0.1917845, -0.2001071, 0.2422187, -0.3795529, 0.3918916

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8492783, upper bound: 1.8609583
time: 2.15 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8492783, upper bound: 1.8618168
time: 2.15 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1806060, 0.8531412, -0.1306743, 0.7579499, -0.9385560, 0.9838154
1: -0.2773540, 0.3141760, -0.2285620, 0.2761554, -0.5535095, 0.5427380
2: -0.3637846, 0.3703282, -0.3156646, 0.3206358, -0.6844203, 0.6859929
3: -0.2560428, 0.2108136, -0.2240667, 0.1623892, -0.4184320, 0.4348803
4: -0.2693838, 0.3510133, -0.2175044, 0.3151521, -0.5845360, 0.5685177
5: -0.4080175, 0.4540177, -0.3628212, 0.4003546, -0.8083721, 0.8168389
6: -0.0590689, 1.2809784, 0.0475165, 1.2628757, -1.3219445, 1.2334620
7: -0.3060597, 0.4132388, -0.2542428, 0.3633691, -0.6694288, 0.6674815
8: -0.3291885, 0.3655639, -0.2839868, 0.3059005, -0.6350890, 0.6495507
9: -0.1996495, 0.2433801, -0.1591880, 0.2074531, -0.4071026, 0.4025681

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8541485, upper bound: 1.8884776
time: 2.00 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8541485, upper bound: 1.8884776
time: 2.02 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1472818, 0.8008128, -0.1217815, 0.6773383, -0.8246201, 0.9225943
1: -0.2474676, 0.2875265, -0.1911542, 0.2469305, -0.4943981, 0.4786807
2: -0.3354613, 0.3360834, -0.2792137, 0.2838729, -0.6193342, 0.6152971
3: -0.2364577, 0.1767326, -0.2030807, 0.1224505, -0.3589082, 0.3798133
4: -0.2360632, 0.3269789, -0.1787544, 0.2901190, -0.5261822, 0.5057334
5: -0.3826554, 0.4188182, -0.3280739, 0.3645307, -0.7471861, 0.7468921
6: 0.0025828, 1.2736733, 0.1345455, 1.2634530, -1.2608702, 1.1391279
7: -0.2708085, 0.3814626, -0.2121164, 0.3285020, -0.5993105, 0.5935790
8: -0.3006588, 0.3255523, -0.2534976, 0.2648360, -0.5654948, 0.5790498
9: -0.1685240, 0.2163475, -0.1320065, 0.1881884, -0.3567123, 0.3483540

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8538202, upper bound: 1.8620255
time: 1.80 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8538202, upper bound: 1.8627467
time: 1.75 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.1290609, 0.7743732, -0.2197241, 0.9010219, -1.0300828, 0.9940973
1: -0.2311405, 0.2722027, -0.3088645, 0.3456420, -0.5767825, 0.5810672
2: -0.3202564, 0.3178847, -0.3928846, 0.4111367, -0.7313932, 0.7107693
3: -0.2265249, 0.1558766, -0.2772069, 0.2491153, -0.4756403, 0.4330835
4: -0.2209340, 0.3114398, -0.3066499, 0.3806731, -0.6016070, 0.6180897
5: -0.3699014, 0.3945377, -0.4323313, 0.4979123, -0.8678136, 0.8268690
6: 0.0357372, 1.2593821, -0.1188840, 1.2895670, -1.2538297, 1.3782661
7: -0.2511581, 0.3651014, -0.3478206, 0.4475762, -0.6987343, 0.7129220
8: -0.2827066, 0.3000788, -0.3624573, 0.4095375, -0.6922442, 0.6625360
9: -0.1526023, 0.2034907, -0.2377264, 0.2838125, -0.4364148, 0.4412171

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_A2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8772503, upper bound: 1.8526566
time: 1.80 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8772503, upper bound: 1.8540221
time: 2.40 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.1199748, 0.6871628, -0.2360516, 0.9213381, -1.0413129, 0.9232143
1: -0.1949311, 0.2423474, -0.3221140, 0.3588542, -0.5537853, 0.5644614
2: -0.2821348, 0.2816170, -0.4051805, 0.4282180, -0.7103528, 0.6867975
3: -0.2024080, 0.1200429, -0.2861544, 0.2650071, -0.4674151, 0.4061973
4: -0.1812648, 0.2880423, -0.3222730, 0.3931868, -0.5744516, 0.6103153
5: -0.3326597, 0.3646601, -0.4428855, 0.5166113, -0.8492709, 0.8075457
6: 0.1258899, 1.2612784, -0.1440104, 1.2948549, -1.1689650, 1.4052888
7: -0.2100204, 0.3319268, -0.3651637, 0.4619819, -0.6720022, 0.6970904
8: -0.2535061, 0.2631369, -0.3766791, 0.4280382, -0.6815442, 0.6398159
9: -0.1269139, 0.1866026, -0.2533768, 0.3006897, -0.4276035, 0.4399794

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_A2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8689843, upper bound: 1.8400545
time: 1.82 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8691804, upper bound: 1.8423948
time: 2.26 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.1190621, 0.7088556, -0.1832446, 0.8467579, -0.9658200, 0.8921001
1: -0.2028118, 0.2528833, -0.2773510, 0.3173974, -0.5202091, 0.5302343
2: -0.2920780, 0.2924893, -0.3627326, 0.3739475, -0.6660255, 0.6552220
3: -0.2087669, 0.1277202, -0.2559755, 0.2149226, -0.4236895, 0.3836958
4: -0.1908915, 0.2956613, -0.2702564, 0.3538890, -0.5447805, 0.5659177
5: -0.3418725, 0.3715344, -0.4055400, 0.4582239, -0.8000964, 0.7770744
6: 0.1036389, 1.2595006, -0.0554651, 1.2800367, -1.1763978, 1.3149657
7: -0.2217293, 0.3400527, -0.3096960, 0.4137470, -0.6354763, 0.6497487
8: -0.2618011, 0.2734258, -0.3311962, 0.3695394, -0.6313404, 0.6046220
9: -0.1361624, 0.1907769, -0.2049094, 0.2491823, -0.3853447, 0.3956863

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_A2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8491901, upper bound: 1.8522712
time: 2.20 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8491901, upper bound: 1.8536285
time: 2.08 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.1222738, 0.6131009, -0.1996390, 0.8670249, -0.9892987, 0.8127398
1: -0.1672523, 0.2226339, -0.2906122, 0.3306332, -0.4978855, 0.5132461
2: -0.2513304, 0.2525931, -0.3750900, 0.3910360, -0.6423663, 0.6276831
3: -0.1891952, 0.1023527, -0.2649561, 0.2308652, -0.4200604, 0.3673088
4: -0.1507747, 0.2701142, -0.2858509, 0.3664552, -0.5172300, 0.5559651
5: -0.3031156, 0.3413504, -0.4159358, 0.4769855, -0.7801011, 0.7572862
6: 0.1983947, 1.2662921, -0.0806559, 1.2839589, -1.0855643, 1.3469479
7: -0.1816922, 0.3038312, -0.3271010, 0.4282016, -0.6098938, 0.6309322
8: -0.2285952, 0.2342818, -0.3454542, 0.3881215, -0.6167167, 0.5797360
9: -0.1151881, 0.1749192, -0.2205907, 0.2661330, -0.3813210, 0.3955100

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_A2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8420091, upper bound: 1.8398406
time: 1.84 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8421853, upper bound: 1.8421853
time: 2.12 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1908983, 0.8756229, -0.6408266, 1.4699628, -1.6608611, 1.5164495
1: -0.2876973, 0.3208417, -0.6528313, 0.6626136, -0.9503108, 0.9736730
2: -0.3745519, 0.3795155, -0.7324177, 0.8555449, -1.2300968, 1.1119332
3: -0.2631456, 0.2193505, -0.5431786, 0.6473091, -0.9104548, 0.7625291
4: -0.2811765, 0.3569371, -0.6751659, 0.8704064, -1.1515830, 1.0321031
5: -0.4184634, 0.4617238, -0.8239776, 0.9598182, -1.3782816, 1.2857015
6: -0.0819756, 1.2819579, -0.8017728, 1.5047332, -1.5867088, 2.0837307
7: -0.3160120, 0.4243731, -0.8008890, 0.7960505, -1.1120625, 1.2252622
8: -0.3360884, 0.3748245, -0.7347701, 0.9041102, -1.2401986, 1.1095946
9: -0.2076989, 0.2501320, -0.6323482, 0.7158800, -0.9235790, 0.8824801

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8327507, upper bound: 1.8175619
time: 1.98 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8343167, upper bound: 1.8215325
time: 2.53 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 5.78 seconds
IS_A1_B1_A1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8751320, upper bound: 1.8425451
IS_A1_B1_A1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8785460, upper bound: 1.8531280
IS_A1_B1_A1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8517709, upper bound: 1.8422138
IS_A1_B1_A1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8528454, upper bound: 1.8528165
IS_A1_B1_A1_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8882222, upper bound: 1.8623181
IS_A1_B1_A1_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8882222, upper bound: 1.8627186
IS_A1_B1_A1_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8609757, upper bound: 1.8619508
IS_A1_B1_A1_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8609757, upper bound: 1.8624052
IS_A1_B1_A1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8696515, upper bound: 1.8322504
IS_A1_B1_A1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8740476, upper bound: 1.8407304
IS_A1_B1_A1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8499831, upper bound: 1.8319271
IS_A1_B1_A1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8519780, upper bound: 1.8404128
IS_A1_B1_A1_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8843058, upper bound: 1.8535278
IS_A1_B1_A1_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8843058, upper bound: 1.8541485
IS_A1_B1_A1_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8601397, upper bound: 1.8530939
IS_A1_B1_A1_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8601397, upper bound: 1.8538200
IS_A1_B1_A1_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8623181, upper bound: 1.8899332
IS_A1_B1_A1_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8623181, upper bound: 1.8899332
IS_A1_B1_A1_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8619508, upper bound: 1.8612699
IS_A1_B1_A1_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8619508, upper bound: 1.8623924
IS_A1_B1_A1_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8627186, upper bound: 1.8899332
IS_A1_B1_A1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8627186, upper bound: 1.8899332
IS_A1_B1_A1_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8624052, upper bound: 1.8622863
IS_A1_B1_A1_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8624052, upper bound: 1.8629863
IS_A1_B1_A1_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8613575, upper bound: 1.8796464
IS_A1_B1_A1_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8613575, upper bound: 1.8802744
IS_A1_B1_A1_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8598876, upper bound: 1.8474190
IS_A1_B1_A1_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8598876, upper bound: 1.8499085
IS_A1_B1_A1_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8843420, upper bound: 1.8541970
IS_A1_B1_A1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8843420, upper bound: 1.8546595
IS_A1_B1_A1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8612797, upper bound: 1.8538901
IS_A1_B1_A1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8612797, upper bound: 1.8496140
IS_A1_B1_A2_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8511876, upper bound: 1.8775614
IS_A1_B1_A2_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8528223, upper bound: 1.8851599
IS_A1_B1_A2_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8491535, upper bound: 1.8479453
IS_A1_B1_A2_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8525192, upper bound: 1.8610949
IS_A1_B1_A2_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8781258, upper bound: 1.8613575
IS_A1_B1_A2_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8781258, upper bound: 1.8623144
IS_A1_B1_A2_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8471214, upper bound: 1.8598876
IS_A1_B1_A2_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8471214, upper bound: 1.8618521
IS_A1_B1_A2_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8494285, upper bound: 1.8635489
IS_A1_B1_A2_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8526566, upper bound: 1.8772503
IS_A1_B1_A2_A1_B2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8339739, upper bound: 1.8391627
IS_A1_B1_A2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8437225, upper bound: 1.8403100
IS_A1_B1_A2_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8712049, upper bound: 1.8525946
IS_A1_B1_A2_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8809846, upper bound: 1.8540616
IS_A1_B1_A2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8469727, upper bound: 1.8526206
IS_A1_B1_A2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8469727, upper bound: 1.8537294
IS_A1_B1_A2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8776334, upper bound: 1.8614700
IS_A1_B1_A2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8776334, upper bound: 1.8622780
IS_A1_B1_A2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8492783, upper bound: 1.8609583
IS_A1_B1_A2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8492783, upper bound: 1.8618168
IS_A1_B1_A2_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8541485, upper bound: 1.8884776
IS_A1_B1_A2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8541485, upper bound: 1.8884776
IS_A1_B1_A2_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8538202, upper bound: 1.8620255
IS_A1_B1_A2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8538202, upper bound: 1.8627467
IS_A1_B1_A2_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8772503, upper bound: 1.8526566
IS_A1_B1_A2_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8772503, upper bound: 1.8540221
IS_A1_B1_A2_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8689843, upper bound: 1.8400545
IS_A1_B1_A2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8691804, upper bound: 1.8423948
IS_A1_B1_A2_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8491901, upper bound: 1.8522712
IS_A1_B1_A2_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8491901, upper bound: 1.8536285
IS_A1_B1_A2_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8420091, upper bound: 1.8398406
IS_A1_B1_A2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8421853, upper bound: 1.8421853
IS_A1_B2_B1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8327507, upper bound: 1.8175619
IS_A1_B2_B1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.78
Output dim: 6, lower bound: -1.8343167, upper bound: 1.8215325
IS_A1_B2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8437101, upper bound: 1.8335032
IS_A1_B2_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8841266, upper bound: 1.8443778
IS_A1_B2_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8850170, upper bound: 1.8480956
IS_A1_B2_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8841266, upper bound: 1.8443778
IS_A1_B2_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8850170, upper bound: 1.8480955
IS_A1_B2_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8845056, upper bound: 1.8478598
IS_A1_B2_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8845056, upper bound: 1.8478598
IS_A1_B2_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8848065, upper bound: 1.8494081
IS_A1_B2_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8848065, upper bound: 1.8494081
IS_A1_B2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8749391, upper bound: 1.8238898
IS_A1_B2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8749391, upper bound: 1.8238898
IS_A1_B2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8755558, upper bound: 1.8238689
IS_A1_B2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8755558, upper bound: 1.8238689
IS_A1_B2_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8752946, upper bound: 1.8296678
IS_A1_B2_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8752946, upper bound: 1.8296678
IS_A1_B2_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8752946, upper bound: 1.8309926
IS_A1_B2_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8752946, upper bound: 1.8309926
IS_A1_B2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8521304, upper bound: 1.8167881
IS_A1_B2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8521304, upper bound: 1.8312210
IS_A2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8335032, upper bound: 1.8437101
IS_A2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8335032, upper bound: 1.8437101
IS_A2_A1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8443778, upper bound: 1.8841266
IS_A2_A1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8480956, upper bound: 1.8850170
IS_A2_A1_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8443778, upper bound: 1.8841266
IS_A2_A1_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8480956, upper bound: 1.8850170
IS_A2_A1_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8467583, upper bound: 1.8858963
IS_A2_A1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8467583, upper bound: 1.8858963
IS_A2_A1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8488624, upper bound: 1.8861909
IS_A2_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 6, lower bound: -1.8488624, upper bound: 1.8861909
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.094357967376709
rel_dist={6: [-1.9135725282356488, 1.9135718444666665]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 189

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8916922, upper bound: 1.8639348
time: 3.15 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8637098, upper bound: 1.8637097
time: 2.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 5.80 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 5.80
Output dim: 6, lower bound: -1.8916922, upper bound: 1.8639348
IS_A2, status: Status.UNKNOWN, split count: 1, time: 5.80
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

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
time: 2.94 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8612994, upper bound: 1.8300520
time: 2.50 seconds

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

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 168

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8434120, upper bound: 1.8297824
time: 2.08 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8297503, upper bound: 1.8297503
time: 1.97 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.19 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 5.19
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 5.19
Output dim: 6, lower bound: -1.8612994, upper bound: 1.8300520
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 5.19
Output dim: 6, lower bound: -1.8434120, upper bound: 1.8297824
IS_A2_A2, status: Status.VERIFIED, split count: 2, time: 5.19
Output dim: 6, lower bound: -1.8297503, upper bound: 1.8297503

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.2336920, 0.9276679, -0.4077451, 1.2271204, -1.4608124, 1.3354130
1: -0.3219900, 0.3554728, -0.4846753, 0.4986582, -0.8206482, 0.8401481
2: -0.4061271, 0.4243887, -0.5721409, 0.6110175, -1.0171446, 0.9965297
3: -0.2861868, 0.2612684, -0.3927653, 0.4440010, -0.7301878, 0.6540337
4: -0.3218213, 0.3896247, -0.5047001, 0.5578194, -0.8796407, 0.8943248
5: -0.4451328, 0.5103453, -0.6008465, 0.7199819, -1.1651148, 1.1111917
6: -0.1471975, 1.2969497, -0.4862225, 1.4411615, -1.5883591, 1.7831722
7: -0.3616745, 0.4618227, -0.5604192, 0.6238955, -0.9855700, 1.0222420
8: -0.3729174, 0.4232192, -0.5457388, 0.6317784, -1.0046959, 0.9689580
9: -0.2493206, 0.2945979, -0.4230016, 0.4714570, -0.7207776, 0.7175995

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8622186, upper bound: 1.8290884
time: 2.47 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
time: 2.70 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.2360516, 0.9213381, -0.3391498, 1.1047845, -1.3408360, 1.2604879
1: -0.3221140, 0.3588542, -0.4180301, 0.4416270, -0.7637410, 0.7768844
2: -0.4051805, 0.4282180, -0.5049126, 0.5322866, -0.9374672, 0.9331306
3: -0.2861544, 0.2650071, -0.3492922, 0.3676039, -0.6537583, 0.6142993
4: -0.3222730, 0.3931868, -0.4320533, 0.4726011, -0.7948741, 0.8252401
5: -0.4428855, 0.5166113, -0.5323161, 0.6358327, -1.0787182, 1.0489273
6: -0.1440104, 1.2948549, -0.3475126, 1.3837935, -1.5278039, 1.6423675
7: -0.3651637, 0.4619819, -0.4778503, 0.5604187, -0.9255824, 0.9398322
8: -0.3766791, 0.4280382, -0.4704264, 0.5455489, -0.9222280, 0.8984646
9: -0.2533768, 0.3006897, -0.3496245, 0.3969221, -0.6502990, 0.6503142

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8579487, upper bound: 1.8287593
time: 3.00 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8612994, upper bound: 1.8300520
time: 3.14 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -0.8361796, 1.6916373, -0.3782749, 1.1809021, -2.0170817, 2.0699122
1: -0.7908013, 0.7925454, -0.4566278, 0.4754850, -1.2662864, 1.2491732
2: -0.8674313, 1.0445075, -0.5464179, 0.5717729, -1.4392042, 1.5909255
3: -0.6669739, 0.8051766, -0.3732519, 0.4125818, -1.0795557, 1.1784285
4: -0.8164440, 1.1000400, -0.4763324, 0.5156271, -1.3320711, 1.5763724
5: -1.0171378, 1.1558790, -0.5687980, 0.6838543, -1.7009921, 1.7246771
6: -1.0718563, 1.6044312, -0.4318709, 1.4234797, -2.4953361, 2.0363021
7: -0.9843407, 0.9428054, -0.5252811, 0.5982280, -1.5825686, 1.4680865
8: -0.8967382, 1.1181587, -0.5158603, 0.5920585, -1.4887967, 1.6340190
9: -0.7806149, 0.9022418, -0.3915270, 0.4355803, -1.2161952, 1.2937688

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 168

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412604, upper bound: 1.8286572
time: 2.42 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8434084, upper bound: 1.8297824
time: 2.04 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.62 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 6, lower bound: -1.8622186, upper bound: 1.8290884
IS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 6, lower bound: -1.8579487, upper bound: 1.8287593
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 6, lower bound: -1.8612994, upper bound: 1.8300520
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 6, lower bound: -1.8412604, upper bound: 1.8286572
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 6, lower bound: -1.8434084, upper bound: 1.8297824

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.3578017, 1.1505049, -1.3887740, 1.3085963
1: -0.3297816, 0.3564569, -0.4384367, 0.4563085, -0.7860901, 0.7948936
2: -0.4150444, 0.4270655, -0.5275022, 0.5501261, -0.9651704, 0.9545677
3: -0.2916944, 0.2636827, -0.3624331, 0.3870284, -0.6787228, 0.6261158
4: -0.3300874, 0.3899961, -0.4547935, 0.4903031, -0.8203905, 0.8447895
5: -0.4548751, 0.5089113, -0.5529270, 0.6561784, -1.1110535, 1.0618383
6: -0.1680785, 1.3109281, -0.3955092, 1.4138204, -1.5818989, 1.7064373
7: -0.3655823, 0.4693472, -0.4995477, 0.5802073, -0.9457897, 0.9688949
8: -0.3765907, 0.4246833, -0.4953920, 0.5659776, -0.9425683, 0.9200754
9: -0.2510976, 0.2932430, -0.3666548, 0.4119251, -0.6630226, 0.6598978

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 189

## Relational analysis of IS_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8622186, upper bound: 1.8290884
time: 2.13 seconds

## Relational analysis of IS_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8622186, upper bound: 1.8290884
time: 2.50 seconds

## BFS IS instance: IS_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.4046560, 1.2226121, -1.4277365, 1.2961564
1: -0.2987980, 0.3325009, -0.4817867, 0.4962968, -0.7950948, 0.8142875
2: -0.3846641, 0.3945331, -0.5694313, 0.6069851, -0.9916492, 0.9639645
3: -0.2706087, 0.2334781, -0.3904986, 0.4407750, -0.7113838, 0.6239767
4: -0.2944325, 0.3679506, -0.5018574, 0.5531512, -0.8475838, 0.8698080
5: -0.4267558, 0.4781152, -0.5976180, 0.7159364, -1.1426922, 1.0757332
6: -0.1026776, 1.2858521, -0.4805676, 1.4396272, -1.5423048, 1.7664196
7: -0.3313026, 0.4365475, -0.5568249, 0.6212189, -0.9525214, 0.9933724
8: -0.3483621, 0.3910702, -0.5427763, 0.6271727, -0.9755348, 0.9338466
9: -0.2218433, 0.2653263, -0.4196700, 0.4677564, -0.6895997, 0.6849963

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 189

## Relational analysis of IS_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
time: 2.34 seconds

## Relational analysis of IS_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
time: 2.72 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -0.2948729, 1.0365634, -1.2931185, 1.2693131
1: -0.3448424, 0.3715456, -0.3790523, 0.4012321, -0.7460746, 0.7505980
2: -0.4289511, 0.4466715, -0.4620942, 0.4858472, -0.9147984, 0.9087657
3: -0.3015637, 0.2813208, -0.3248466, 0.3166918, -0.6182555, 0.6061674
4: -0.3472072, 0.4048828, -0.3859903, 0.4336113, -0.7808185, 0.7908731
5: -0.4669372, 0.5324135, -0.4985565, 0.5759596, -1.0428967, 1.0309700
6: -0.1975362, 1.3229021, -0.2685531, 1.3585694, -1.5561056, 1.5914552
7: -0.3851250, 0.4853222, -0.4250308, 0.5216075, -0.9067325, 0.9103530
8: -0.3923242, 0.4465339, -0.4302375, 0.4893626, -0.8816868, 0.8767714
9: -0.2674738, 0.3125915, -0.3001776, 0.3486806, -0.6161544, 0.6127691

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 189

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8579487, upper bound: 1.8287593
time: 2.55 seconds

## Relational analysis of IS_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8579487, upper bound: 1.8287594
time: 2.60 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.2067769, 0.8844675, -0.3367072, 1.1009101, -1.3076869, 1.2211747
1: -0.2982824, 0.3353040, -0.4158249, 0.4393273, -0.7376097, 0.7511289
2: -0.3830329, 0.3976636, -0.5025038, 0.5297366, -0.9127696, 0.9001673
3: -0.2701186, 0.2365419, -0.3479493, 0.3646058, -0.6347244, 0.5844912
4: -0.2941514, 0.3709590, -0.4294414, 0.4697236, -0.7638750, 0.8004004
5: -0.4239522, 0.4835648, -0.5303960, 0.6325419, -1.0564941, 1.0139608
6: -0.0985513, 1.2864097, -0.3428971, 1.3823812, -1.4809325, 1.6293068
7: -0.3340706, 0.4360552, -0.4747277, 0.5582772, -0.8923478, 0.9107830
8: -0.3514939, 0.3950947, -0.4679361, 0.5424806, -0.8939744, 0.8630308
9: -0.2252686, 0.2706831, -0.3467785, 0.3942468, -0.6195154, 0.6174616

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8488611, upper bound: 1.8166423
time: 2.40 seconds

## Relational analysis of IS_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8494511, upper bound: 1.8181667
time: 2.10 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.8261355, 1.6957741, -0.3335419, 1.1103027, -1.9364381, 2.0293159
1: -0.7845350, 0.7813374, -0.4162333, 0.4338018, -1.2183368, 1.1975707
2: -0.8624935, 1.0301049, -0.5033086, 0.5248662, -1.3873596, 1.5334134
3: -0.6554159, 0.7940063, -0.3487143, 0.3578281, -1.0132440, 1.1427207
4: -0.8107981, 1.0799718, -0.4286669, 0.4632168, -1.2740148, 1.5086387
5: -1.0105178, 1.1361258, -0.5331617, 0.6234801, -1.6339979, 1.6692874
6: -1.0676947, 1.6187458, -0.3488653, 1.3970141, -2.4647088, 1.9676111
7: -0.9711751, 0.9359220, -0.4690937, 0.5586243, -1.5297995, 1.4050157
8: -0.8822247, 1.1008432, -0.4704472, 0.5355538, -1.4177785, 1.5712904
9: -0.7688699, 0.8852291, -0.3393127, 0.3857433, -1.1546133, 1.2245418

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8407838, upper bound: 1.8286488
time: 1.81 seconds

## Relational analysis of IS_A2_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8412186, upper bound: 1.8286488
time: 2.05 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.7844378, 1.6371753, -0.3754041, 1.1767790, -1.9612168, 2.0125794
1: -0.7545584, 0.7572792, -0.4539109, 0.4731984, -1.2277567, 1.2111901
2: -0.8325227, 0.9933141, -0.5440305, 0.5687660, -1.4012887, 1.5373447
3: -0.6313741, 0.7623650, -0.3716356, 0.4093971, -1.0407712, 1.1340005
4: -0.7793881, 1.0365195, -0.4735236, 0.5116554, -1.2910435, 1.5100431
5: -0.9667646, 1.1034745, -0.5659974, 0.6805803, -1.6473448, 1.6694719
6: -1.0031563, 1.5844967, -0.4268457, 1.4219713, -2.4251275, 2.0113425
7: -0.9345270, 0.9043884, -0.5221443, 0.5957885, -1.5303156, 1.4265326
8: -0.8537888, 1.0604813, -0.5130845, 0.5885241, -1.4423130, 1.5735657
9: -0.7392236, 0.8510032, -0.3883551, 0.4319663, -1.1711899, 1.2393583

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 168

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8430532, upper bound: 1.8297824
time: 2.18 seconds

## Relational analysis of IS_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8433551, upper bound: 1.8297824
time: 2.05 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.38 seconds
IS_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 6, lower bound: -1.8622186, upper bound: 1.8290884
IS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 6, lower bound: -1.8622186, upper bound: 1.8290884
IS_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
IS_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
IS_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 6, lower bound: -1.8579487, upper bound: 1.8287593
IS_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 6, lower bound: -1.8579487, upper bound: 1.8287594
IS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 6, lower bound: -1.8488611, upper bound: 1.8166423
IS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 6, lower bound: -1.8494511, upper bound: 1.8181667
IS_A2_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 5.38
Output dim: 6, lower bound: -1.8407838, upper bound: 1.8286488
IS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 6, lower bound: -1.8412186, upper bound: 1.8286488
IS_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 6, lower bound: -1.8430532, upper bound: 1.8297824
IS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 6, lower bound: -1.8433551, upper bound: 1.8297824

## BFS IS instance: IS_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.3305279, 1.1073103, -1.3455794, 1.2813225
1: -0.3297816, 0.3564569, -0.4140316, 0.4306878, -0.7604694, 0.7704885
2: -0.4150444, 0.4270655, -0.5009556, 0.5214322, -0.9364765, 0.9280211
3: -0.2916944, 0.2636827, -0.3473814, 0.3539606, -0.6456550, 0.6110641
4: -0.3300874, 0.3899961, -0.4259421, 0.4602557, -0.7903430, 0.8159381
5: -0.4548751, 0.5089113, -0.5314786, 0.6183604, -1.0732355, 1.0403899
6: -0.1680785, 1.3109281, -0.3451698, 1.3970816, -1.5651602, 1.6560979
7: -0.3655823, 0.4693472, -0.4653145, 0.5564607, -0.9220431, 0.9346617
8: -0.3765907, 0.4246833, -0.4684740, 0.5311835, -0.9077742, 0.8931574
9: -0.2510976, 0.2932430, -0.3357365, 0.3816711, -0.6327687, 0.6289794

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8622186, upper bound: 1.8290764
time: 2.31 seconds

## Relational analysis of IS_A1_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8617109, upper bound: 1.8290755
time: 1.98 seconds

## BFS IS instance: IS_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.7435856, 1.8219769, -2.0602460, 1.6943802
1: -0.3297816, 0.3564569, -0.7961034, 0.8118749, -1.1416565, 1.1525604
2: -0.4150444, 0.4270655, -0.9185721, 0.9481149, -1.3631592, 1.3456376
3: -0.2916944, 0.2636827, -0.5808446, 0.8549458, -1.1466403, 0.8445274
4: -0.3300874, 0.3899961, -0.8763115, 0.9339783, -1.2640657, 1.2663076
5: -0.4548751, 0.5089113, -0.8759522, 1.1576483, -1.6125233, 1.3848634
6: -0.1680785, 1.3109281, -1.1632617, 1.7168252, -1.8849038, 2.4741898
7: -0.3655823, 0.4693472, -0.9896207, 0.9251333, -1.2907157, 1.4589679
8: -0.3765907, 0.4246833, -0.9035685, 1.0405298, -1.4171206, 1.3282518
9: -0.2510976, 0.2932430, -0.8096569, 0.8203414, -1.0714390, 1.1028998

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8622186, upper bound: 1.8290884
time: 2.07 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8622186, upper bound: 1.8290884
time: 2.55 seconds

## BFS IS instance: IS_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.3722159, 1.1733804, -1.3785048, 1.2637162
1: -0.2987980, 0.3325009, -0.4514494, 0.4698666, -0.7686647, 0.7839503
2: -0.3846641, 0.3945331, -0.5415025, 0.5651585, -0.9498225, 0.9360356
3: -0.2706087, 0.2334781, -0.3701423, 0.4050531, -0.6756618, 0.6036204
4: -0.2944325, 0.3679506, -0.4703737, 0.5072894, -0.8017219, 0.8383242
5: -0.4267558, 0.4781152, -0.5640730, 0.6753143, -1.1020701, 1.0421882
6: -0.1026776, 1.2858521, -0.4225191, 1.4219546, -1.5246322, 1.7083712
7: -0.3313026, 0.4365475, -0.5180095, 0.5930730, -0.9243755, 0.9545571
8: -0.3483621, 0.3910702, -0.5102229, 0.5839884, -0.9323505, 0.9012930
9: -0.2218433, 0.2653263, -0.3840388, 0.4276934, -0.6495367, 0.6493651

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_A2_B1_A1

### Relational analysis result of IS_A1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
time: 2.17 seconds

## Relational analysis of IS_A1_A1_A2_B1_A2

### Relational analysis result of IS_A1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8642123, upper bound: 1.8300602
time: 2.06 seconds

## BFS IS instance: IS_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.8881479, 1.9902647, -2.1953890, 1.7796483
1: -0.2987980, 0.3325009, -0.9448962, 0.8592061, -1.1580040, 1.2773970
2: -0.3846641, 0.3945331, -1.0060918, 1.2324014, -1.6170654, 1.4006250
3: -0.2706087, 0.2334781, -0.7503050, 0.9411615, -1.2117703, 0.9837831
4: -0.2944325, 0.3679506, -0.9584443, 1.2703116, -1.5647441, 1.3263948
5: -0.4267558, 0.4781152, -1.1230767, 1.3313054, -1.7580612, 1.6011919
6: -0.1026776, 1.2858521, -1.4070342, 1.7503563, -1.8530340, 2.6928864
7: -0.3313026, 0.4365475, -1.1175296, 1.0484126, -1.3797151, 1.5540771
8: -0.3483621, 0.3910702, -1.0226146, 1.3370495, -1.6854116, 1.4136848
9: -0.2218433, 0.2653263, -0.9352762, 1.0318565, -1.2536998, 1.2006024

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_A1_A2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
time: 2.57 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
time: 2.58 seconds

## BFS IS instance: IS_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -0.2682301, 1.0023152, -1.2588704, 1.2426703
1: -0.3448424, 0.3715456, -0.3574942, 0.3795063, -0.7243487, 0.7290398
2: -0.4289511, 0.4466715, -0.4420578, 0.4577690, -0.8867201, 0.8887292
3: -0.3015637, 0.2813208, -0.3102405, 0.2907727, -0.5923364, 0.5915613
4: -0.3472072, 0.4048828, -0.3605982, 0.4128413, -0.7600485, 0.7654810
5: -0.4669372, 0.5324135, -0.4800309, 0.5445315, -1.0114686, 1.0124444
6: -0.1975362, 1.3229021, -0.2269817, 1.3429223, -1.5404584, 1.5498838
7: -0.3851250, 0.4853222, -0.3966308, 0.4982494, -0.8833743, 0.8819530
8: -0.3923242, 0.4465339, -0.4073272, 0.4587462, -0.8510704, 0.8538611
9: -0.2674738, 0.3125915, -0.2749172, 0.3209248, -0.5883986, 0.5875088

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_A1_B1_A1

### Relational analysis result of IS_A1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8579487, upper bound: 1.8287377
time: 1.84 seconds

## Relational analysis of IS_A1_A2_A1_B1_A2

### Relational analysis result of IS_A1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8576959, upper bound: 1.8287378
time: 2.17 seconds

## BFS IS instance: IS_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -0.6801782, 1.5888977, -1.8454528, 1.6546185
1: -0.3448424, 0.3715456, -0.7029074, 0.7040936, -1.0489360, 1.0744530
2: -0.4289511, 0.4466715, -0.7661029, 0.8836422, -1.3125933, 1.2127744
3: -0.3015637, 0.2813208, -0.5416631, 0.6870940, -0.9886578, 0.8229839
4: -0.3472072, 0.4048828, -0.7663231, 0.7178132, -1.0650203, 1.1712059
5: -0.4669372, 0.5324135, -0.7886119, 0.9940000, -1.4609371, 1.3210254
6: -0.1975362, 1.3229021, -0.9150661, 1.6601486, -1.8576849, 2.2379682
7: -0.3851250, 0.4853222, -0.8327870, 0.8690823, -1.2542074, 1.3181092
8: -0.3923242, 0.4465339, -0.7798890, 0.9140267, -1.3063509, 1.2264228
9: -0.2674738, 0.3125915, -0.6645848, 0.7292258, -0.9966996, 0.9771763

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8453049, upper bound: 1.8153099
time: 2.49 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8455265, upper bound: 1.8168335
time: 9.58 seconds

## BFS IS instance: IS_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1418619, 0.7892814, -0.1491887, 0.8177679, -0.9596297, 0.9384701
1: -0.2417822, 0.2838043, -0.2520266, 0.2876110, -0.5293932, 0.5358309
2: -0.3298238, 0.3312632, -0.3414498, 0.3372473, -0.6670710, 0.6727130
3: -0.2331933, 0.1712702, -0.2406450, 0.1748348, -0.4080280, 0.4119152
4: -0.2305267, 0.3234906, -0.2410817, 0.3277267, -0.5582534, 0.5645722
5: -0.3773848, 0.4141394, -0.3907229, 0.4210179, -0.7984028, 0.8048624
6: 0.0155045, 1.2737827, -0.0105105, 1.2794653, -1.2639608, 1.2842932
7: -0.2653973, 0.3759036, -0.2715290, 0.3861742, -0.6515715, 0.6474326
8: -0.2965988, 0.3189987, -0.3042389, 0.3250157, -0.6216145, 0.6232376
9: -0.1643996, 0.2140305, -0.1656019, 0.2164815, -0.3808811, 0.3796324

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 189

## Relational analysis of IS_A1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8488611, upper bound: 1.8166423
time: 1.97 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8488611, upper bound: 1.8166423
time: 2.23 seconds

## BFS IS instance: IS_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1484677, 0.8021158, -0.2146879, 0.9131356, -1.0616033, 1.0168037
1: -0.2484228, 0.2886999, -0.3089058, 0.3397292, -0.5881520, 0.5976057
2: -0.3363540, 0.3374221, -0.3950753, 0.4043389, -0.7406929, 0.7324974
3: -0.2370916, 0.1783540, -0.2777402, 0.2413937, -0.4784852, 0.4560943
4: -0.2369839, 0.3281827, -0.3049134, 0.3757891, -0.6127731, 0.6330962
5: -0.3834973, 0.4207538, -0.4373106, 0.4912487, -0.8747460, 0.8580644
6: 0.0008506, 1.2755382, -0.1257447, 1.3008263, -1.2999758, 1.4012829
7: -0.2722592, 0.3823006, -0.3411181, 0.4465334, -0.7187926, 0.7234187
8: -0.3020198, 0.3276488, -0.3595527, 0.4026453, -0.7046652, 0.6872016
9: -0.1698050, 0.2174858, -0.2274627, 0.2734749, -0.4432799, 0.4449485

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 189

## Relational analysis of IS_A1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8494511, upper bound: 1.8181667
time: 2.37 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2

### Relational analysis result of IS_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8494511, upper bound: 1.8181667
time: 1.95 seconds

## BFS IS instance: IS_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.7979034, 1.6659223, -0.2903454, 1.0461423, -1.8440456, 1.9562677
1: -0.7648128, 0.7621134, -0.3790317, 0.3957989, -1.1606116, 1.1411451
2: -0.8435066, 1.0021583, -0.4637430, 0.4798131, -1.3233197, 1.4659013
3: -0.6380908, 0.7706629, -0.3251740, 0.3104502, -0.9485410, 1.0958369
4: -0.7906191, 1.0453122, -0.3845334, 0.4286603, -1.2192795, 1.4298456
5: -0.9830699, 1.1076347, -0.5010560, 0.5683812, -1.5514511, 1.6086906
6: -1.0303032, 1.6076341, -0.2745431, 1.3720298, -2.4023330, 1.8821771
7: -0.9440065, 0.9151468, -0.4194821, 0.5206435, -1.4646499, 1.3346289
8: -0.8588787, 1.0694300, -0.4323633, 0.4825815, -1.3414602, 1.5017934
9: -0.7462275, 0.8573418, -0.2924274, 0.3398126, -1.0860401, 1.1497692

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_A1_A1_B2_A1

### Relational analysis result of IS_A2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8276821, upper bound: 1.8165530
time: 2.00 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8288798, upper bound: 1.8167540
time: 2.01 seconds

## BFS IS instance: IS_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.7409655, 1.5911280, -0.2292824, 0.9530854, -1.6940509, 1.8204104
1: -0.7240844, 0.7275990, -0.3259403, 0.3487006, -1.0727849, 1.0535393
2: -0.8031114, 0.9502400, -0.4131392, 0.4173312, -1.2204425, 1.3633792
3: -0.6045082, 0.7263724, -0.2893661, 0.2525430, -0.8570512, 1.0157385
4: -0.7482117, 0.9830473, -0.3230298, 0.3843424, -1.1325541, 1.3060771
5: -0.9244519, 1.0594876, -0.4554485, 0.5033277, -1.4277797, 1.5149361
6: -0.9453980, 1.5674840, -0.1661356, 1.3251669, -2.2705650, 1.7336197
7: -0.8926209, 0.8721398, -0.3549941, 0.4638858, -1.3565066, 1.2271338
8: -0.8177298, 1.0119834, -0.3741209, 0.4161291, -1.2338588, 1.3861043
9: -0.7043506, 0.8079602, -0.2362507, 0.2815626, -0.9859132, 1.0442109

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_A1_A2_B1_A1

### Relational analysis result of IS_A2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8295363, upper bound: 1.8177303
time: 2.41 seconds

## Relational analysis of IS_A2_A1_A2_B1_A2

### Relational analysis result of IS_A2_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8311200, upper bound: 1.8179027
time: 2.47 seconds

## BFS IS instance: IS_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.7562270, 1.6073766, -0.3321052, 1.1072761, -1.8635031, 1.9394817
1: -0.7347767, 0.7380106, -0.4146990, 0.4327310, -1.1675078, 1.1527097
2: -0.8134547, 0.9653451, -0.5015900, 0.5236222, -1.3370769, 1.4669352
3: -0.6139329, 0.7389856, -0.3479337, 0.3559262, -0.9698591, 1.0869193
4: -0.7591577, 1.0017658, -0.4267391, 0.4625704, -1.2217281, 1.4285049
5: -0.9393134, 1.0749592, -0.5318785, 0.6228238, -1.5621372, 1.6068377
6: -0.9657427, 1.5735791, -0.3457068, 1.3962330, -2.3619757, 1.9192858
7: -0.9073047, 0.8834643, -0.4673346, 0.5571554, -1.4644601, 1.3507988
8: -0.8304140, 1.0290151, -0.4689128, 0.5343155, -1.3647294, 1.4979279
9: -0.7165381, 0.8230456, -0.3372369, 0.3845990, -1.1011372, 1.1602825

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_A1_A2_B2_A1

### Relational analysis result of IS_A2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8298208, upper bound: 1.8177303
time: 2.39 seconds

## Relational analysis of IS_A2_A1_A2_B2_A2

### Relational analysis result of IS_A2_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8313504, upper bound: 1.8179027
time: 2.82 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.38 seconds
IS_A1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8622186, upper bound: 1.8290764
IS_A1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8617109, upper bound: 1.8290755
IS_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8622186, upper bound: 1.8290884
IS_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8622186, upper bound: 1.8290884
IS_A1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
IS_A1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8642123, upper bound: 1.8300602
IS_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
IS_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8643217, upper bound: 1.8300619
IS_A1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8579487, upper bound: 1.8287377
IS_A1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8576959, upper bound: 1.8287378
IS_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8453049, upper bound: 1.8153099
IS_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8455265, upper bound: 1.8168335
IS_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8488611, upper bound: 1.8166423
IS_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8488611, upper bound: 1.8166423
IS_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8494511, upper bound: 1.8181667
IS_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8494511, upper bound: 1.8181667
IS_A2_A1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8276821, upper bound: 1.8165530
IS_A2_A1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8288798, upper bound: 1.8167540
IS_A2_A1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8295363, upper bound: 1.8177303
IS_A2_A1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8311200, upper bound: 1.8179027
IS_A2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8298208, upper bound: 1.8177303
IS_A2_A1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.38
Output dim: 6, lower bound: -1.8313504, upper bound: 1.8179027

## BFS IS instance: IS_A1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1169841, 0.7567631, -0.3062389, 1.0690753, -1.1860595, 1.0630021
1: -0.2204840, 0.2508239, -0.3924268, 0.4087634, -0.6292474, 0.6432507
2: -0.3103864, 0.2950570, -0.4773085, 0.4960929, -0.8064792, 0.7723655
3: -0.2196565, 0.1302018, -0.3340923, 0.3257217, -0.5453782, 0.4642941
4: -0.2076873, 0.2957779, -0.4003425, 0.4404673, -0.6481546, 0.6961204
5: -0.3641408, 0.3731782, -0.5127494, 0.5861509, -0.9502918, 0.8859277
6: 0.0626922, 1.2523227, -0.3019156, 1.3826998, -1.3200077, 1.5542383
7: -0.2253656, 0.3574407, -0.4365337, 0.5351175, -0.7604831, 0.7939745
8: -0.2659138, 0.2762619, -0.4466663, 0.5006104, -0.7665241, 0.7229283
9: -0.1328410, 0.1881934, -0.3085958, 0.3553117, -0.4881527, 0.4967891

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8955489, upper bound: 1.8875233
time: 2.61 seconds

## Relational analysis of IS_A1_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8955489, upper bound: 1.8875233
time: 2.04 seconds

## BFS IS instance: IS_A1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1993656, 0.9015930, -0.3145869, 1.0818012, -1.2811668, 1.2161798
1: -0.2981919, 0.3253166, -0.3996075, 0.4162969, -0.7144888, 0.7249242
2: -0.3856547, 0.3865064, -0.4852809, 0.5048045, -0.8904592, 0.8717873
3: -0.2705576, 0.2259107, -0.3386689, 0.3347931, -0.6053506, 0.5645796
4: -0.2927791, 0.3607083, -0.4088223, 0.4472878, -0.7400669, 0.7695305
5: -0.4300287, 0.4653581, -0.5189545, 0.5972765, -1.0273052, 0.9843127
6: -0.1069707, 1.2916774, -0.3166493, 1.3877221, -1.4946928, 1.6083267
7: -0.3243314, 0.4349272, -0.4461933, 0.5424551, -0.8667865, 0.8811204
8: -0.3430020, 0.3811920, -0.4540452, 0.5111333, -0.8541353, 0.8352373
9: -0.2137607, 0.2537034, -0.3178598, 0.3641723, -0.5779330, 0.5715632

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8836525, upper bound: 1.8740243
time: 2.16 seconds

## Relational analysis of IS_A1_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8840255, upper bound: 1.8766958
time: 2.48 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.6045691, 1.5491773, -1.7874464, 1.5553637
1: -0.3297816, 0.3564569, -0.6613263, 0.6880524, -1.0178341, 1.0177832
2: -0.4150444, 0.4270655, -0.7709063, 0.8063177, -1.2213621, 1.1979718
3: -0.2916944, 0.2636827, -0.4965475, 0.6931372, -0.9848316, 0.7602302
4: -0.3300874, 0.3899961, -0.7208173, 0.7805793, -1.1106668, 1.1108134
5: -0.4548751, 0.5089113, -0.7483861, 0.9799878, -1.4348629, 1.2572974
6: -0.1680785, 1.3109281, -0.8638911, 1.5669374, -1.7350160, 2.1748192
7: -0.3655823, 0.4693472, -0.8170909, 0.7957259, -1.1613083, 1.2864380
8: -0.3765907, 0.4246833, -0.7467324, 0.8722093, -1.2488000, 1.1714157
9: -0.2510976, 0.2932430, -0.6605608, 0.6806305, -0.9317281, 0.9538038

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8609331, upper bound: 1.8290755
time: 1.97 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8617109, upper bound: 1.8290755
time: 4.40 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.6103973, 1.5564686, -1.7947377, 1.5611919
1: -0.3297816, 0.3564569, -0.6663769, 0.6939174, -1.0236990, 1.0228338
2: -0.4150444, 0.4270655, -0.7759100, 0.8131639, -1.2282083, 1.2029755
3: -0.2916944, 0.2636827, -0.4997124, 0.6998774, -0.9915718, 0.7633951
4: -0.3300874, 0.3899961, -0.7260563, 0.7880439, -1.1181312, 1.1160524
5: -0.4548751, 0.5089113, -0.7529014, 0.9907979, -1.4456730, 1.2618127
6: -0.1680785, 1.3109281, -0.8736334, 1.5693415, -1.7374201, 2.1845615
7: -0.3655823, 0.4693472, -0.8244229, 0.8005888, -1.1661712, 1.2937701
8: -0.3765907, 0.4246833, -0.7517854, 0.8809407, -1.2575314, 1.1764687
9: -0.2510976, 0.2932430, -0.6659949, 0.6880851, -0.9391826, 0.9592379

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8609331, upper bound: 1.8290755
time: 2.16 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8617109, upper bound: 1.8290755
time: 2.45 seconds

## BFS IS instance: IS_A1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1159616, 0.7033735, -0.3478231, 1.1341276, -1.2500892, 1.0511966
1: -0.2010893, 0.2369420, -0.4293826, 0.4469394, -0.6480287, 0.6663246
2: -0.2882367, 0.2758111, -0.5175823, 0.5397190, -0.8279557, 0.7933934
3: -0.2049150, 0.1182289, -0.3567882, 0.3749268, -0.5798417, 0.4750172
4: -0.1859916, 0.2844449, -0.4440683, 0.4786170, -0.6646086, 0.7285131
5: -0.3422161, 0.3617626, -0.5448487, 0.6426256, -0.9848417, 0.9066113
6: 0.1144768, 1.2558804, -0.3763925, 1.4071257, -1.2926489, 1.6322728
7: -0.2056796, 0.3379471, -0.4868948, 0.5713810, -0.7770605, 0.8248419
8: -0.2516249, 0.2579985, -0.4850380, 0.5534283, -0.8050532, 0.7430365
9: -0.1192000, 0.1819619, -0.3552544, 0.4010786, -0.5202786, 0.5372163

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8846882, upper bound: 1.8745727
time: 2.43 seconds

## Relational analysis of IS_A1_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8849655, upper bound: 1.8773454
time: 2.52 seconds

## BFS IS instance: IS_A1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1622779, 0.8381748, -0.3562621, 1.1476963, -1.3099742, 1.1944370
1: -0.2638702, 0.2978253, -0.4370032, 0.4548587, -0.7187289, 0.7348285
2: -0.3524194, 0.3495883, -0.5258554, 0.5485228, -0.9009423, 0.8754438
3: -0.2470456, 0.1912882, -0.3614275, 0.3852326, -0.6322781, 0.5527157
4: -0.2530104, 0.3358105, -0.4530852, 0.4885149, -0.7415253, 0.7888957
5: -0.3999231, 0.4305325, -0.5515293, 0.6539692, -1.0538924, 0.9820617
6: -0.0352900, 1.2740173, -0.3924366, 1.4123473, -1.4476373, 1.6664538
7: -0.2856037, 0.3983221, -0.4976377, 0.5787803, -0.8643840, 0.8959599
8: -0.3113918, 0.3422747, -0.4936590, 0.5640182, -0.8754100, 0.8359337
9: -0.1802069, 0.2218284, -0.3650150, 0.4102674, -0.5904743, 0.5868434

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8838999, upper bound: 1.8745717
time: 2.29 seconds

## Relational analysis of IS_A1_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8841459, upper bound: 1.8773447
time: 3.08 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.7173703, 1.6870488, -1.8921732, 1.6088706
1: -0.2987980, 0.3325009, -0.7764640, 0.7331138, -1.0319118, 1.1089649
2: -0.3846641, 0.3945331, -0.8442040, 1.0165434, -1.4012074, 1.2387371
3: -0.2706087, 0.2334781, -0.6208835, 0.7689818, -1.0395906, 0.8543617
4: -0.2944325, 0.3679506, -0.7919869, 1.0278358, -1.3222684, 1.1599374
5: -0.4267558, 0.4781152, -0.9307849, 1.1166674, -1.5434232, 1.4089001
6: -0.1026776, 1.2858521, -1.0574242, 1.5987313, -1.7014089, 2.3432763
7: -0.3313026, 0.4365475, -0.9222007, 0.8927671, -1.2240696, 1.3587482
8: -0.3483621, 0.3910702, -0.8443253, 1.0920823, -1.4404444, 1.2353956
9: -0.2218433, 0.2653263, -0.7626777, 0.8414990, -1.0633423, 1.0280039

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8514642, upper bound: 1.8166531
time: 2.11 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8522350, upper bound: 1.8181751
time: 2.39 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.7234483, 1.6940877, -1.8992121, 1.6149486
1: -0.2987980, 0.3325009, -0.7820697, 0.7384813, -1.0372794, 1.1145706
2: -0.3846641, 0.3945331, -0.8491067, 1.0247238, -1.4093878, 1.2436398
3: -0.2706087, 0.2334781, -0.6252918, 0.7753217, -1.0459304, 0.8587700
4: -0.2944325, 0.3679506, -0.7969993, 1.0370207, -1.3314532, 1.1649499
5: -0.4267558, 0.4781152, -0.9366944, 1.1275326, -1.5542884, 1.4148096
6: -0.1026776, 1.2858521, -1.0677397, 1.6004364, -1.7031140, 2.3535919
7: -0.3313026, 0.4365475, -0.9293858, 0.8981249, -1.2294275, 1.3659333
8: -0.3483621, 0.3910702, -0.8527557, 1.1020607, -1.4504228, 1.2438259
9: -0.2218433, 0.2653263, -0.7681721, 0.8496748, -1.0715181, 1.0334984

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8514642, upper bound: 1.8166531
time: 2.19 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8522350, upper bound: 1.8181751
time: 2.55 seconds

## BFS IS instance: IS_A1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1348380, 0.7975615, -0.2439169, 0.9710965, -1.1059345, 1.0414784
1: -0.2396843, 0.2733411, -0.3378217, 0.3599654, -0.5996498, 0.6111628
2: -0.3301149, 0.3207297, -0.4238153, 0.4323848, -0.7624997, 0.7445450
3: -0.2327745, 0.1562737, -0.2970951, 0.2671152, -0.4998898, 0.4533688
4: -0.2288631, 0.3139742, -0.3373208, 0.3944353, -0.6232984, 0.6512951
5: -0.3808719, 0.3988070, -0.4641170, 0.5172175, -0.8980893, 0.8629240
6: 0.0152110, 1.2646937, -0.1890393, 1.3298299, -1.3146189, 1.4537331
7: -0.2537427, 0.3744118, -0.3708082, 0.4768237, -0.7305664, 0.7452200
8: -0.2880276, 0.3033789, -0.3864518, 0.4314448, -0.7194723, 0.6898307
9: -0.1522545, 0.2038378, -0.2515131, 0.2960493, -0.4483037, 0.4553509

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_A1_B1_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8762696, upper bound: 1.8738951
time: 2.71 seconds

## Relational analysis of IS_A1_A2_A1_B1_A1_B2

### Relational analysis result of IS_A1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8765197, upper bound: 1.8765327
time: 2.64 seconds

## BFS IS instance: IS_A1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2179237, 0.9248135, -0.2522867, 0.9819235, -1.1998472, 1.1771002
1: -0.3134779, 0.3405452, -0.3446018, 0.3667017, -0.6801797, 0.6851470
2: -0.3998227, 0.4063864, -0.4301178, 0.4411296, -0.8409523, 0.8365042
3: -0.2805924, 0.2438077, -0.3016328, 0.2752517, -0.5558441, 0.5454405
4: -0.3101083, 0.3756746, -0.3453138, 0.4007972, -0.7109056, 0.7209884
5: -0.4418477, 0.4890904, -0.4695652, 0.5266973, -0.9685450, 0.9586555
6: -0.1369656, 1.3034592, -0.2021481, 1.3344164, -1.4713820, 1.5056072
7: -0.3441505, 0.4511290, -0.3797017, 0.4842016, -0.8283521, 0.8308306
8: -0.3590224, 0.4031833, -0.3936640, 0.4408714, -0.7998939, 0.7968473
9: -0.2303850, 0.2731903, -0.2595258, 0.3046210, -0.5350059, 0.5327162

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_A1_B1_A2_A1

### Relational analysis result of IS_A1_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8735649, upper bound: 1.8761157
time: 3.03 seconds

## Relational analysis of IS_A1_A2_A1_B1_A2_A2

### Relational analysis result of IS_A1_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8756542, upper bound: 1.8765338
time: 2.37 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.1896633, 0.8795776, -0.4972652, 1.3189930, -1.5086563, 1.3768427
1: -0.2881022, 0.3192363, -0.5477046, 0.5604128, -0.8485150, 0.8669409
2: -0.3753560, 0.3779203, -0.6199721, 0.6947278, -1.0700837, 0.9978924
3: -0.2635004, 0.2176148, -0.4376862, 0.5117170, -0.7752173, 0.6553010
4: -0.2812157, 0.3554775, -0.5848498, 0.5825380, -0.8637537, 0.9403273
5: -0.4201623, 0.4594539, -0.6477416, 0.7941395, -1.2143018, 1.1071955
6: -0.0842916, 1.2847902, -0.6028616, 1.5045694, -1.5888610, 1.8876518
7: -0.3146171, 0.4242426, -0.6393007, 0.7034785, -1.0180956, 1.0635433
8: -0.3349793, 0.3727897, -0.6113641, 0.7119550, -1.0469342, 0.9841539
9: -0.2055180, 0.2473785, -0.4931862, 0.5490233, -0.7545412, 0.7405646

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_A1_B2_B1_A1

### Relational analysis result of IS_A1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8453049, upper bound: 1.8152803
time: 2.96 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_A2

### Relational analysis result of IS_A1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8451291, upper bound: 1.8152810
time: 1.94 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.1983991, 0.8930646, -0.5608325, 1.4096783, -1.6080775, 1.4538970
1: -0.2959167, 0.3258590, -0.6010351, 0.6106313, -0.9065480, 0.9268941
2: -0.3828703, 0.3867720, -0.6699769, 0.7606162, -1.1434865, 1.0567490
3: -0.2687544, 0.2257566, -0.4733876, 0.5730258, -0.8417802, 0.6991442
4: -0.2900515, 0.3618659, -0.6473859, 0.6297904, -0.9198418, 1.0092518
5: -0.4267098, 0.4689465, -0.6954200, 0.8639656, -1.2906754, 1.1643665
6: -0.1005784, 1.2900848, -0.7089144, 1.5548033, -1.6553817, 1.9989992
7: -0.3238592, 0.4324284, -0.7067699, 0.7605828, -1.0844420, 1.1391983
8: -0.3425427, 0.3823935, -0.6688023, 0.7824288, -1.1249715, 1.0511959
9: -0.2131679, 0.2555539, -0.5533780, 0.6123431, -0.8255111, 0.8089319

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_A1_B2_B2_A1

### Relational analysis result of IS_A1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8455265, upper bound: 1.8168124
time: 2.31 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_A2

### Relational analysis result of IS_A1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8453206, upper bound: 1.8168106
time: 2.33 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.1418619, 0.7892814, -0.1315861, 0.7875699, -0.9294317, 0.9208675
1: -0.2417822, 0.2838043, -0.2351888, 0.2708704, -0.5126527, 0.5189931
2: -0.3298238, 0.3312632, -0.3259391, 0.3173927, -0.6472164, 0.6572024
3: -0.2331933, 0.1712702, -0.2303033, 0.1516239, -0.3848171, 0.4015735
4: -0.2305267, 0.3234906, -0.2240233, 0.3128085, -0.5433352, 0.5475138
5: -0.3773848, 0.4141394, -0.3771718, 0.3984623, -0.7758471, 0.7913113
6: 0.0155045, 1.2737827, 0.0256221, 1.2714212, -1.2559166, 1.2481606
7: -0.2653973, 0.3759036, -0.2490303, 0.3704776, -0.6358748, 0.6249338
8: -0.2965988, 0.3189987, -0.2874089, 0.3009290, -0.5975278, 0.6064076
9: -0.1643996, 0.2140305, -0.1488544, 0.2036574, -0.3680571, 0.3628850

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452576, upper bound: 1.8137280
time: 2.71 seconds

## Relational analysis of IS_A1_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8488611, upper bound: 1.8166423
time: 1.94 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.1418619, 0.7892814, -0.4553567, 1.3499527, -1.4918146, 1.2446381
1: -0.2417822, 0.2838043, -0.5419130, 0.5319115, -0.7736937, 0.8257173
2: -0.3298238, 0.3312632, -0.6124225, 0.6320112, -0.9618350, 0.9436858
3: -0.2331933, 0.1712702, -0.4050403, 0.5508766, -0.7840699, 0.5763105
4: -0.2305267, 0.3234906, -0.5192004, 0.5546509, -0.7851776, 0.8426909
5: -0.3773848, 0.4141394, -0.6630939, 0.7329614, -1.1103462, 1.0772333
6: 0.0155045, 1.2737827, -0.6166009, 1.5098497, -1.4943452, 1.8903836
7: -0.2653973, 0.3759036, -0.6097134, 0.6576377, -0.9230349, 0.9856170
8: -0.2965988, 0.3189987, -0.5670747, 0.7592266, -1.0558254, 0.8860734
9: -0.1643996, 0.2140305, -0.4687383, 0.3887703, -0.5531698, 0.6827689

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452576, upper bound: 1.8137280
time: 2.36 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8488611, upper bound: 1.8166423
time: 1.89 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.1484677, 0.8021158, -0.1883273, 0.8800330, -1.0285007, 0.9904431
1: -0.2484228, 0.2886999, -0.2875983, 0.3183219, -0.5667447, 0.5762982
2: -0.3363540, 0.3374221, -0.3751941, 0.3767065, -0.7130605, 0.7126162
3: -0.2370916, 0.1783540, -0.2631949, 0.2157769, -0.4528684, 0.4415489
4: -0.2369839, 0.3281827, -0.2799078, 0.3552566, -0.5922405, 0.6080905
5: -0.3834973, 0.4207538, -0.4204234, 0.4601977, -0.8436950, 0.8411772
6: 0.0008506, 1.2755382, -0.0846785, 1.2893199, -1.2884693, 1.3602166
7: -0.2722592, 0.3823006, -0.3130104, 0.4234391, -0.6956983, 0.6953110
8: -0.3020198, 0.3276488, -0.3359196, 0.3723456, -0.6743654, 0.6635685
9: -0.1698050, 0.2174858, -0.2025242, 0.2459956, -0.4158006, 0.4200100

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_A2_A2_B2_B1_A1

### Relational analysis result of IS_A1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8458858, upper bound: 1.8153799
time: 2.46 seconds

## Relational analysis of IS_A1_A2_A2_B2_B1_A2

### Relational analysis result of IS_A1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8494507, upper bound: 1.8181667
time: 1.88 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.1484677, 0.8021158, -0.6029408, 1.4631039, -1.6115716, 1.4050566
1: -0.2484228, 0.2886999, -0.6351355, 0.6448317, -0.8932545, 0.9238354
2: -0.3363540, 0.3374221, -0.7014237, 0.8048807, -1.1412346, 1.0388458
3: -0.2370916, 0.1783540, -0.4961900, 0.6140727, -0.8511643, 0.6745440
4: -0.2369839, 0.3281827, -0.6877422, 0.6622455, -0.8992294, 1.0159249
5: -0.3834973, 0.4207538, -0.7253211, 0.9125700, -1.2960674, 1.1460750
6: 0.0008506, 1.2755382, -0.7736180, 1.5817307, -1.5808802, 2.0491562
7: -0.2722592, 0.3823006, -0.7515774, 0.7975643, -1.0698235, 1.1338780
8: -0.3020198, 0.3276488, -0.7043688, 0.8304054, -1.1324252, 1.0320177
9: -0.1698050, 0.2174858, -0.5938302, 0.6560583, -0.8258633, 0.8113160

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8354233, upper bound: 1.8015677
time: 2.46 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2_B2

### Relational analysis result of IS_A1_A2_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8382346, upper bound: 1.8062330
time: 2.60 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.24 seconds
IS_A1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8955489, upper bound: 1.8875233
IS_A1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8955489, upper bound: 1.8875233
IS_A1_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8836525, upper bound: 1.8740243
IS_A1_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8840255, upper bound: 1.8766958
IS_A1_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8609331, upper bound: 1.8290755
IS_A1_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8617109, upper bound: 1.8290755
IS_A1_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8609331, upper bound: 1.8290755
IS_A1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8617109, upper bound: 1.8290755
IS_A1_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8846882, upper bound: 1.8745727
IS_A1_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8849655, upper bound: 1.8773454
IS_A1_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8838999, upper bound: 1.8745717
IS_A1_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8841459, upper bound: 1.8773447
IS_A1_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8514642, upper bound: 1.8166531
IS_A1_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8522350, upper bound: 1.8181751
IS_A1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8514642, upper bound: 1.8166531
IS_A1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8522350, upper bound: 1.8181751
IS_A1_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8762696, upper bound: 1.8738951
IS_A1_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8765197, upper bound: 1.8765327
IS_A1_A2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8735649, upper bound: 1.8761157
IS_A1_A2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8756542, upper bound: 1.8765338
IS_A1_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8453049, upper bound: 1.8152803
IS_A1_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8451291, upper bound: 1.8152810
IS_A1_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8455265, upper bound: 1.8168124
IS_A1_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8453206, upper bound: 1.8168106
IS_A1_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8452576, upper bound: 1.8137280
IS_A1_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8488611, upper bound: 1.8166423
IS_A1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8452576, upper bound: 1.8137280
IS_A1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8488611, upper bound: 1.8166423
IS_A1_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8458858, upper bound: 1.8153799
IS_A1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8494507, upper bound: 1.8181667
IS_A1_A2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8354233, upper bound: 1.8015677
IS_A1_A2_A2_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 6.24
Output dim: 6, lower bound: -1.8382346, upper bound: 1.8062330

## BFS IS instance: IS_A1_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1169841, 0.7567631, -0.1654937, 0.8432150, -0.9601991, 0.9222568
1: -0.2204840, 0.2508239, -0.2668080, 0.3006066, -0.5210906, 0.5176319
2: -0.3103864, 0.2950570, -0.3549408, 0.3533347, -0.6637211, 0.6499978
3: -0.2196565, 0.1302018, -0.2491131, 0.1950061, -0.4146626, 0.3793149
4: -0.2076873, 0.2957779, -0.2568765, 0.3375055, -0.5451928, 0.5526544
5: -0.3641408, 0.3731782, -0.4022612, 0.4325969, -0.7967377, 0.7754394
6: 0.0626922, 1.2523227, -0.0412191, 1.2728709, -1.2101787, 1.2935417
7: -0.2253656, 0.3574407, -0.2892045, 0.4018666, -0.6272321, 0.6466452
8: -0.2659138, 0.2762619, -0.3137623, 0.3459979, -0.6119117, 0.5900242
9: -0.1328410, 0.1881934, -0.1841531, 0.2243565, -0.3571974, 0.3723465

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8834210, upper bound: 1.8763758
time: 2.45 seconds

## Relational analysis of IS_A1_A1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8848933, upper bound: 1.8766947
time: 2.27 seconds

## BFS IS instance: IS_A1_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1169841, 0.7567631, -0.1685483, 0.8370934, -0.9540775, 0.9253114
1: -0.2204840, 0.2508239, -0.2673145, 0.3045498, -0.5250338, 0.5181384
2: -0.3103864, 0.2950570, -0.3542798, 0.3579172, -0.6683036, 0.6493368
3: -0.2196565, 0.1302018, -0.2491835, 0.1993806, -0.4190371, 0.3793853
4: -0.2076873, 0.2957779, -0.2578110, 0.3416699, -0.5493572, 0.5535889
5: -0.3641408, 0.3731782, -0.3999754, 0.4398440, -0.8039849, 0.7731537
6: 0.0626922, 1.2523227, -0.0392712, 1.2765508, -1.2138586, 1.2915938
7: -0.2253656, 0.3574407, -0.2934108, 0.4024189, -0.6277845, 0.6508515
8: -0.2659138, 0.2762619, -0.3182402, 0.3517555, -0.6176693, 0.5945022
9: -0.1328410, 0.1881934, -0.1888067, 0.2312129, -0.3640539, 0.3770000

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_A1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8834210, upper bound: 1.8763758
time: 2.00 seconds

## Relational analysis of IS_A1_A1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_A1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8848933, upper bound: 1.8766947
time: 3.43 seconds

## BFS IS instance: IS_A1_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1403547, 0.8122416, -0.1387052, 0.8150617, -0.9554164, 0.9509468
1: -0.2460097, 0.2780254, -0.2455928, 0.2743976, -0.5204072, 0.5236182
2: -0.3361274, 0.3263265, -0.3373495, 0.3230408, -0.6591682, 0.6636760
3: -0.2369108, 0.1633222, -0.2375122, 0.1561762, -0.3930870, 0.4008344
4: -0.2357173, 0.3174095, -0.2346119, 0.3161458, -0.5518632, 0.5520214
5: -0.3871087, 0.4025793, -0.3894457, 0.4029427, -0.7900513, 0.7920250
6: 0.0011004, 1.2622731, -0.0001883, 1.2739650, -1.2728646, 1.2624614
7: -0.2607962, 0.3805324, -0.2554582, 0.3807014, -0.6414976, 0.6359906
8: -0.2910372, 0.3088648, -0.2929691, 0.3065656, -0.5976028, 0.6018339
9: -0.1574453, 0.2058766, -0.1512799, 0.2050320, -0.3624773, 0.3571565

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8694154, upper bound: 1.8636618
time: 2.72 seconds

## Relational analysis of IS_A1_A1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_A1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8738198, upper bound: 1.8646525
time: 2.42 seconds

## BFS IS instance: IS_A1_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1464125, 0.8241361, -0.1943346, 0.9036708, -1.0500833, 1.0184708
1: -0.2521383, 0.2828651, -0.2964672, 0.3210162, -0.5731544, 0.5793324
2: -0.3420772, 0.3323510, -0.3850463, 0.3812687, -0.7233459, 0.7173973
3: -0.2406743, 0.1704648, -0.2694058, 0.2193801, -0.4600544, 0.4398705
4: -0.2416758, 0.3219216, -0.2889079, 0.3578139, -0.5994897, 0.6108295
5: -0.3924929, 0.4090801, -0.4309185, 0.4633895, -0.8558823, 0.8399986
6: -0.0125664, 1.2654368, -0.1073317, 1.3023412, -1.3149076, 1.3727684
7: -0.2677125, 0.3863299, -0.3182203, 0.4322104, -0.6999229, 0.7045502
8: -0.2961976, 0.3168942, -0.3419449, 0.3767642, -0.6729617, 0.6588392
9: -0.1628464, 0.2093654, -0.2045393, 0.2469522, -0.4097987, 0.4139047

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_A1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8840255, upper bound: 1.8766958
time: 2.65 seconds

## Relational analysis of IS_A1_A1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8840255, upper bound: 1.8766958
time: 2.57 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.2150804, 0.9212438, -0.4529352, 1.2927241, -1.5078045, 1.3741790
1: -0.3109617, 0.3378776, -0.5218340, 0.5471938, -0.8581554, 0.8597116
2: -0.3975527, 0.4028647, -0.6183964, 0.6492516, -1.0468043, 1.0212611
3: -0.2791124, 0.2411552, -0.4113983, 0.5089200, -0.7880324, 0.6525534
4: -0.3078786, 0.3725088, -0.5561133, 0.6047690, -0.9126476, 0.9286221
5: -0.4400503, 0.4828545, -0.6227210, 0.7797796, -1.2198299, 1.1055754
6: -0.1316835, 1.2991772, -0.5682919, 1.4535425, -1.5852261, 1.8674691
7: -0.3409778, 0.4488276, -0.6244746, 0.6608884, -1.0018662, 1.0733023
8: -0.3565967, 0.3987262, -0.5882289, 0.6841519, -1.0407486, 0.9869551
9: -0.2288533, 0.2696380, -0.4862385, 0.5182737, -0.7471271, 0.7558765

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8635653, upper bound: 1.8341833
time: 1.91 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8638894, upper bound: 1.8351324
time: 2.46 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.2237346, 0.9321672, -0.5606544, 1.4785151, -1.7022498, 1.4928217
1: -0.3179772, 0.3448228, -0.6216136, 0.6466639, -0.9646411, 0.9664364
2: -0.4040706, 0.4119112, -0.7276643, 0.7605605, -1.1646311, 1.1395755
3: -0.2838015, 0.2495669, -0.4726355, 0.6391031, -0.9229045, 0.7222024
4: -0.3161477, 0.3790524, -0.6736431, 0.7287970, -1.0449446, 1.0526955
5: -0.4455283, 0.4926402, -0.7128924, 0.9211846, -1.3667128, 1.2055326
6: -0.1452520, 1.3034813, -0.7814059, 1.5384302, -1.6836822, 2.0848873
7: -0.3501682, 0.4564676, -0.7609833, 0.7572580, -1.1074262, 1.2174509
8: -0.3640393, 0.4084372, -0.7022231, 0.8171774, -1.1812167, 1.1106603
9: -0.2371470, 0.2784684, -0.6090840, 0.6325469, -0.8696939, 0.8875524

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8640798, upper bound: 1.8335440
time: 2.60 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8643846, upper bound: 1.8346781
time: 1.95 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.2150804, 0.9212438, -0.4694827, 1.3168366, -1.5319170, 1.3907266
1: -0.3109617, 0.3378776, -0.5364997, 0.5630803, -0.8740420, 0.8743774
2: -0.3975527, 0.4028647, -0.6339178, 0.6672000, -1.0647527, 1.0367825
3: -0.2791124, 0.2411552, -0.4202351, 0.5287945, -0.8079069, 0.6613903
4: -0.3078786, 0.3725088, -0.5728511, 0.6247543, -0.9326329, 0.9453598
5: -0.4400503, 0.4828545, -0.6355826, 0.8046046, -1.2446549, 1.1184371
6: -0.1316835, 1.2991772, -0.5984606, 1.4635689, -1.5952525, 1.8976377
7: -0.3409778, 0.4488276, -0.6454326, 0.6748934, -1.0158713, 1.0942602
8: -0.3565967, 0.3987262, -0.6041974, 0.7062442, -1.0628409, 1.0029237
9: -0.2288533, 0.2696380, -0.5042855, 0.5373365, -0.7661898, 0.7739235

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8481146, upper bound: 1.8156140
time: 2.13 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8484632, upper bound: 1.8171526
time: 2.21 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.2237346, 0.9321672, -0.5661469, 1.4849336, -1.7086682, 1.4983141
1: -0.3179772, 0.3448228, -0.6261758, 0.6522018, -0.9701790, 0.9709986
2: -0.4040706, 0.4119112, -0.7321571, 0.7669899, -1.1710606, 1.1440684
3: -0.2838015, 0.2495669, -0.4753654, 0.6453622, -0.9291637, 0.7249323
4: -0.3161477, 0.3790524, -0.6783838, 0.7358412, -1.0519888, 1.0574362
5: -0.4455283, 0.4926402, -0.7170714, 0.9314083, -1.3769366, 1.2097116
6: -0.1452520, 1.3034813, -0.7900602, 1.5406256, -1.6858776, 2.0935416
7: -0.3501682, 0.4564676, -0.7677506, 0.7616212, -1.1117895, 1.2242182
8: -0.3640393, 0.4084372, -0.7067519, 0.8254877, -1.1895270, 1.1151891
9: -0.2371470, 0.2784684, -0.6141170, 0.6395921, -0.8767391, 0.8925853

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8489993, upper bound: 1.8156140
time: 2.50 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8493331, upper bound: 1.8171503
time: 2.26 seconds

## BFS IS instance: IS_A1_A1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1142331, 0.6248693, -0.1615430, 0.8518895, -0.9661226, 0.7864124
1: -0.1684797, 0.2171838, -0.2667814, 0.2953595, -0.4638391, 0.4839653
2: -0.2546882, 0.2329551, -0.3568579, 0.3478458, -0.6025341, 0.5898130
3: -0.1785344, 0.1017102, -0.2500576, 0.1860972, -0.3646316, 0.3517678
4: -0.1497392, 0.2627160, -0.2557500, 0.3349824, -0.4847215, 0.5184659
5: -0.3125792, 0.3281232, -0.4063123, 0.4307032, -0.7432824, 0.7344356
6: 0.1964228, 1.2484608, -0.0451713, 1.2841892, -1.0877664, 1.2936320
7: -0.1719384, 0.3055758, -0.2836190, 0.4006270, -0.5725654, 0.5891948
8: -0.2208437, 0.2172101, -0.3135061, 0.3383982, -0.5592419, 0.5307162
9: -0.1093071, 0.1639370, -0.1736806, 0.2207869, -0.3300940, 0.3376176

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_A1_A2_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8846882, upper bound: 1.8745727
time: 4.44 seconds

## Relational analysis of IS_A1_A1_A2_B1_A1_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8846882, upper bound: 1.8745727
time: 2.48 seconds

## BFS IS instance: IS_A1_A1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1143816, 0.6295999, -0.2276203, 0.9446074, -1.0589890, 0.8572202
1: -0.1714844, 0.2188726, -0.3231135, 0.3480085, -0.5194929, 0.5419861
2: -0.2574211, 0.2358201, -0.4097951, 0.4161761, -0.6735973, 0.6456152
3: -0.1808843, 0.1025696, -0.2873053, 0.2518540, -0.4327383, 0.3898749
4: -0.1523274, 0.2644736, -0.3203205, 0.3835465, -0.5358738, 0.5847940
5: -0.3147168, 0.3311387, -0.4517156, 0.5020185, -0.8167353, 0.7828543
6: 0.1903963, 1.2491183, -0.1586750, 1.3180195, -1.1276232, 1.4077933
7: -0.1741680, 0.3086150, -0.3537360, 0.4611502, -0.6353182, 0.6623509
8: -0.2231580, 0.2199273, -0.3702211, 0.4147900, -0.6379480, 0.5901484
9: -0.1095108, 0.1650383, -0.2363999, 0.2816265, -0.3911372, 0.4014382

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_A1_A2_B1_A1_B2_B1

### Relational analysis result of IS_A1_A1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8849655, upper bound: 1.8773454
time: 2.11 seconds

## Relational analysis of IS_A1_A1_A2_B1_A1_B2_B2

### Relational analysis result of IS_A1_A1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8849655, upper bound: 1.8773454
time: 2.16 seconds

## BFS IS instance: IS_A1_A1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1176215, 0.7494462, -0.1680191, 0.8615530, -0.9791745, 0.9174652
1: -0.2182280, 0.2551163, -0.2724917, 0.3007829, -0.5190110, 0.5276079
2: -0.3084475, 0.2986492, -0.3622220, 0.3542982, -0.6627457, 0.6608712
3: -0.2188092, 0.1326183, -0.2533581, 0.1941375, -0.4129467, 0.3859763
4: -0.2057186, 0.2996725, -0.2613320, 0.3399032, -0.5456219, 0.5610046
5: -0.3604629, 0.3799571, -0.4110089, 0.4376380, -0.7981009, 0.7909660
6: 0.0669345, 1.2618383, -0.0570672, 1.2867854, -1.2198509, 1.3189056
7: -0.2283913, 0.3552113, -0.2907562, 0.4062555, -0.6346469, 0.6459675
8: -0.2707058, 0.2807751, -0.3186597, 0.3477404, -0.6184462, 0.5994348
9: -0.1358775, 0.1927442, -0.1803405, 0.2246379, -0.3605154, 0.3730847

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_A1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8838999, upper bound: 1.8745717
time: 2.42 seconds

## Relational analysis of IS_A1_A1_A2_B1_A2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8838999, upper bound: 1.8745717
time: 2.27 seconds

## BFS IS instance: IS_A1_A1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1200023, 0.7586315, -0.2355450, 0.9548503, -1.0748526, 0.9941765
1: -0.2220147, 0.2592478, -0.3295538, 0.3543732, -0.5763879, 0.5888017
2: -0.3127536, 0.3030805, -0.4157969, 0.4244473, -0.7372010, 0.7188774
3: -0.2215446, 0.1368611, -0.2916258, 0.2595482, -0.4810928, 0.4284869
4: -0.2098281, 0.3028935, -0.3278986, 0.3895608, -0.5993889, 0.6307920
5: -0.3643635, 0.3833872, -0.4568354, 0.5109798, -0.8753433, 0.8402227
6: 0.0572097, 1.2629392, -0.1711748, 1.3221567, -1.2649469, 1.4341140
7: -0.2330560, 0.3589402, -0.3621590, 0.4681654, -0.7012215, 0.7210991
8: -0.2745536, 0.2852304, -0.3770764, 0.4237082, -0.6982619, 0.6623068
9: -0.1394567, 0.1947418, -0.2439656, 0.2897201, -0.4291768, 0.4387074

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_A1_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8841459, upper bound: 1.8773447
time: 2.57 seconds

## Relational analysis of IS_A1_A1_A2_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8841459, upper bound: 1.8773447
time: 2.54 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.1417945, 0.7986491, -0.4846644, 1.3121879, -1.4539825, 1.2833135
1: -0.2436263, 0.2820661, -0.5515579, 0.5587511, -0.8023774, 0.8336240
2: -0.3326283, 0.3298559, -0.6316955, 0.7172972, -1.0499254, 0.9615514
3: -0.2347354, 0.1685201, -0.4468477, 0.5298030, -0.7645384, 0.6153678
4: -0.2328690, 0.3215658, -0.5705556, 0.6862406, -0.9191096, 0.8921214
5: -0.3814673, 0.4105566, -0.6752958, 0.8201414, -1.2016087, 1.0858524
6: 0.0093051, 1.2688062, -0.6066937, 1.4397334, -1.4304283, 1.8755000
7: -0.2639399, 0.3781549, -0.6533630, 0.6850510, -0.9489908, 1.0315180
8: -0.2951243, 0.3155533, -0.6132805, 0.7520310, -1.0471553, 0.9288338
9: -0.1618411, 0.2112207, -0.5181344, 0.5727223, -0.7345634, 0.7293550

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8509735, upper bound: 1.8185044
time: 2.15 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8533099, upper bound: 1.8227900
time: 2.45 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.1474488, 0.8081646, -0.5646544, 1.4368458, -1.5842946, 1.3728189
1: -0.2489926, 0.2864673, -0.6283389, 0.6189813, -0.8679739, 0.9148062
2: -0.3377682, 0.3351870, -0.7038172, 0.8207753, -1.1585435, 1.0390042
3: -0.2377780, 0.1751053, -0.5063580, 0.6125413, -0.8503193, 0.6814633
4: -0.2380448, 0.3256819, -0.6460720, 0.8049315, -1.0429763, 0.9717539
5: -0.3860441, 0.4162297, -0.7623248, 0.9225960, -1.3086400, 1.1785545
6: -0.0019987, 1.2701281, -0.7588896, 1.4891969, -1.4911957, 2.0290177
7: -0.2699518, 0.3833180, -0.7460646, 0.7559304, -1.0258822, 1.1293826
8: -0.2995212, 0.3233469, -0.6927519, 0.8696819, -1.1692030, 1.0160987
9: -0.1670614, 0.2143013, -0.6032155, 0.6662367, -0.8332981, 0.8175167

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8513032, upper bound: 1.8202194
time: 2.35 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8536409, upper bound: 1.8241040
time: 1.95 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.1417945, 0.7986491, -0.4898330, 1.3189511, -1.4607457, 1.2884822
1: -0.2436263, 0.2820661, -0.5564890, 0.5633915, -0.8070178, 0.8385551
2: -0.3326283, 0.3298559, -0.6361877, 0.7243409, -1.0569692, 0.9660437
3: -0.2347354, 0.1685201, -0.4507134, 0.5352615, -0.7699968, 0.6192334
4: -0.2328690, 0.3215658, -0.5749099, 0.6940025, -0.9268715, 0.8964757
5: -0.3814673, 0.4105566, -0.6804597, 0.8299574, -1.2114247, 1.0910163
6: 0.0093051, 1.2688062, -0.6160907, 1.4421929, -1.4328878, 1.8848969
7: -0.2639399, 0.3781549, -0.6597201, 0.6893654, -0.9533053, 1.0378749
8: -0.2951243, 0.3155533, -0.6208766, 0.7606805, -1.0558047, 0.9364299
9: -0.1618411, 0.2112207, -0.5226662, 0.5798352, -0.7416763, 0.7338869

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_A1_A2_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8482545, upper bound: 1.8141332
time: 1.87 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B1_A2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8514642, upper bound: 1.8166531
time: 2.84 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.1474488, 0.8081646, -0.5695743, 1.4433751, -1.5908239, 1.3777390
1: -0.2489926, 0.2864673, -0.6329781, 0.6234310, -0.8724235, 0.9194455
2: -0.3377682, 0.3351870, -0.7081178, 0.8273454, -1.1651136, 1.0433048
3: -0.2377780, 0.1751053, -0.5100542, 0.6176116, -0.8553896, 0.6851596
4: -0.2380448, 0.3256819, -0.6501709, 0.8121234, -1.0501683, 0.9758527
5: -0.3860441, 0.4162297, -0.7672011, 0.9320189, -1.3180630, 1.1834308
6: -0.0019987, 1.2701281, -0.7678003, 1.4917847, -1.4937835, 2.0379286
7: -0.2699518, 0.3833180, -0.7519708, 0.7602106, -1.0301623, 1.1352887
8: -0.2995212, 0.3233469, -0.7000757, 0.8778432, -1.1773643, 1.0234226
9: -0.1670614, 0.2143013, -0.6072765, 0.6728771, -0.8399385, 0.8215778

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_A1_A2_B2_B2_B2_A1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8489977, upper bound: 1.8157347
time: 2.05 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B2_A2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8522350, upper bound: 1.8181751
time: 2.61 seconds

## BFS IS instance: IS_A1_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1145320, 0.7094084, -0.1156584, 0.7145571, -0.8290891, 0.8250668
1: -0.2033171, 0.2367926, -0.2058134, 0.2376900, -0.4410071, 0.4426060
2: -0.2905828, 0.2751794, -0.2934054, 0.2745970, -0.5651798, 0.5685848
3: -0.2061004, 0.1188367, -0.2075004, 0.1200003, -0.3261006, 0.3263371
4: -0.1878069, 0.2841798, -0.1897118, 0.2851441, -0.4729511, 0.4738916
5: -0.3455643, 0.3611054, -0.3485991, 0.3631186, -0.7086829, 0.7097045
6: 0.1088573, 1.2535816, 0.1033517, 1.2580874, -1.1492300, 1.1502299
7: -0.2061361, 0.3398477, -0.2064821, 0.3424145, -0.5485506, 0.5463298
8: -0.2517108, 0.2572814, -0.2535997, 0.2578214, -0.5095322, 0.5108811
9: -0.1184053, 0.1810046, -0.1170569, 0.1813502, -0.2997555, 0.2980614

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_A2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8610059, upper bound: 1.8633373
time: 2.65 seconds

## Relational analysis of IS_A1_A2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_A2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8667888, upper bound: 1.8644964
time: 2.70 seconds

## BFS IS instance: IS_A1_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1148170, 0.7242222, -0.1357836, 0.8036636, -0.9184806, 0.8600059
1: -0.2090025, 0.2400004, -0.2415256, 0.2726614, -0.4816639, 0.4815260
2: -0.2966937, 0.2817458, -0.3327518, 0.3205708, -0.6172644, 0.6144977
3: -0.2104950, 0.1220360, -0.2343688, 0.1543062, -0.3648012, 0.3564048
4: -0.1940650, 0.2877068, -0.2304778, 0.3142927, -0.5083577, 0.5181846
5: -0.3512622, 0.3664503, -0.3841599, 0.3999001, -0.7511623, 0.7506102
6: 0.0939583, 1.2553573, 0.0098546, 1.2702819, -1.1763237, 1.2455027
7: -0.2118927, 0.3453822, -0.2528373, 0.3765251, -0.5884178, 0.5982195
8: -0.2569899, 0.2637534, -0.2898335, 0.3038847, -0.5608746, 0.5535868
9: -0.1221303, 0.1840075, -0.1503561, 0.2038547, -0.3259850, 0.3343636

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_A2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8757816, upper bound: 1.8756270
time: 2.54 seconds

## Relational analysis of IS_A1_A2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_A2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8765094, upper bound: 1.8765051
time: 2.23 seconds

## BFS IS instance: IS_A1_A2_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.1143923, 0.6762591, -0.1845182, 0.8861129, -1.0005052, 0.8607773
1: -0.1907001, 0.2293597, -0.2870783, 0.3137769, -0.5044770, 0.5164380
2: -0.2771038, 0.2585178, -0.3757042, 0.3716135, -0.6487174, 0.6342220
3: -0.1960519, 0.1115347, -0.2630261, 0.2107401, -0.4067921, 0.3745608
4: -0.1735284, 0.2758650, -0.2784800, 0.3507326, -0.5242610, 0.5543449
5: -0.3336231, 0.3488657, -0.4223424, 0.4527809, -0.7864040, 0.7712080
6: 0.1424876, 1.2517723, -0.0871951, 1.2926211, -1.1501336, 1.3389673
7: -0.1921100, 0.3274561, -0.3082858, 0.4224252, -0.6145351, 0.6357418
8: -0.2395796, 0.2415887, -0.3314750, 0.3661189, -0.6056985, 0.5730637
9: -0.1101430, 0.1736953, -0.1968471, 0.2385152, -0.3486581, 0.3705424

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_A2_A1_B1_A2_A1_A1

### Relational analysis result of IS_A1_A2_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8725772, upper bound: 1.8752054
time: 2.21 seconds

## Relational analysis of IS_A1_A2_A1_B1_A2_A1_A2

### Relational analysis result of IS_A1_A2_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8735507, upper bound: 1.8760842
time: 2.30 seconds

## BFS IS instance: IS_A1_A2_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.1213401, 0.7653091, -0.1927539, 0.8990881, -1.0204282, 0.9580630
1: -0.2244929, 0.2590053, -0.2945601, 0.3199352, -0.5444281, 0.5535654
2: -0.3152511, 0.3037073, -0.3829210, 0.3798222, -0.6950732, 0.6866283
3: -0.2231227, 0.1371470, -0.2679609, 0.2182175, -0.4413402, 0.4051079
4: -0.2121990, 0.3028923, -0.2869235, 0.3566714, -0.5688704, 0.5898159
5: -0.3676960, 0.3833970, -0.4286244, 0.4615340, -0.8292300, 0.8120214
6: 0.0509518, 1.2614751, -0.1026394, 1.2988383, -1.2478864, 1.3641145
7: -0.2340710, 0.3610334, -0.3167411, 0.4302866, -0.6643577, 0.6777745
8: -0.2750545, 0.2855799, -0.3395189, 0.3751019, -0.6501564, 0.6250988
9: -0.1393407, 0.1943661, -0.2038182, 0.2459669, -0.3853076, 0.3981844

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_A2_A1_B1_A2_A2_A1

### Relational analysis result of IS_A1_A2_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8747323, upper bound: 1.8756109
time: 2.56 seconds

## Relational analysis of IS_A1_A2_A1_B1_A2_A2_A2

### Relational analysis result of IS_A1_A2_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8756370, upper bound: 1.8765060
time: 2.56 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1145320, 0.7094084, -0.4732809, 1.2884263, -1.4029583, 1.1826893
1: -0.2033171, 0.2367926, -0.5283677, 0.5411094, -0.7444265, 0.7651603
2: -0.2905828, 0.2751794, -0.6021020, 0.6696699, -0.9602527, 0.8772814
3: -0.2061004, 0.1188367, -0.4247871, 0.4883736, -0.6944740, 0.5436237
4: -0.1878069, 0.2841798, -0.5618823, 0.5643800, -0.7521869, 0.8460621
5: -0.3455643, 0.3611054, -0.6308267, 0.7672533, -1.1128176, 0.9919321
6: 0.1088573, 1.2535816, -0.5658835, 1.4890893, -1.3802319, 1.8194652
7: -0.2061361, 0.3398477, -0.6138877, 0.6824805, -0.8886166, 0.9537354
8: -0.2517108, 0.2572814, -0.5910878, 0.6849810, -0.9366918, 0.8483692
9: -0.1184053, 0.1810046, -0.4699795, 0.5244014, -0.6428068, 0.6509841

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8310373, upper bound: 1.7985680
time: 1.77 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8335583, upper bound: 1.8027418
time: 2.27 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1546477, 0.8331527, -0.4816481, 1.2990829, -1.4537306, 1.3148007
1: -0.2588787, 0.2902472, -0.5351204, 0.5478445, -0.8067232, 0.8253677
2: -0.3482723, 0.3411486, -0.6083404, 0.6784137, -1.0266860, 0.9494890
3: -0.2443664, 0.1806186, -0.4292791, 0.4965096, -0.7408760, 0.6098977
4: -0.2479031, 0.3293337, -0.5698928, 0.5707270, -0.8186301, 0.8992264
5: -0.3972849, 0.4210583, -0.6367406, 0.7766559, -1.1739408, 1.0577989
6: -0.0261015, 1.2726338, -0.5788000, 1.4945207, -1.5206221, 1.8514338
7: -0.2769204, 0.3927052, -0.6227454, 0.6898075, -0.9667280, 1.0154506
8: -0.3050076, 0.3298633, -0.5981506, 0.6944112, -0.9994187, 0.9280139
9: -0.1706091, 0.2160270, -0.4780478, 0.5329933, -0.7036024, 0.6940747

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8307919, upper bound: 1.7985679
time: 2.19 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8332663, upper bound: 1.8027418
time: 2.31 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1148170, 0.7242222, -0.5366991, 1.3788429, -1.4936600, 1.2609212
1: -0.2090025, 0.2400004, -0.5815629, 0.5912064, -0.8002090, 0.8215633
2: -0.2966937, 0.2817458, -0.6519639, 0.7354578, -1.0321515, 0.9337097
3: -0.2104950, 0.1220360, -0.4603986, 0.5495601, -0.7600551, 0.5824346
4: -0.1940650, 0.2877068, -0.6242725, 0.6115152, -0.8055803, 0.9119793
5: -0.3512622, 0.3664503, -0.6783941, 0.8369070, -1.1881692, 1.0448444
6: 0.0939583, 1.2553573, -0.6716070, 1.5390040, -1.4450457, 1.9269643
7: -0.2118927, 0.3453822, -0.6812284, 0.7394212, -0.9513139, 1.0266106
8: -0.2569899, 0.2637534, -0.6483623, 0.7553167, -1.0123066, 0.9121156
9: -0.1221303, 0.1840075, -0.5300947, 0.5875921, -0.7097224, 0.7141021

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A2_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8313310, upper bound: 1.7997585
time: 2.19 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8338186, upper bound: 1.8047472
time: 1.98 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1613360, 0.8456774, -0.5447762, 1.3892395, -1.5505755, 1.3904536
1: -0.2654895, 0.2953812, -0.5881031, 0.5977086, -0.8631981, 0.8834842
2: -0.3546954, 0.3475894, -0.6580277, 0.7438708, -1.0985663, 1.0056171
3: -0.2483198, 0.1884033, -0.4647633, 0.5573920, -0.8057119, 0.6531665
4: -0.2542327, 0.3342273, -0.6320066, 0.6176542, -0.8718869, 0.9662339
5: -0.4031046, 0.4279559, -0.6841259, 0.8460248, -1.2491294, 1.1120818
6: -0.0407807, 1.2761383, -0.6841855, 1.5444298, -1.5852104, 1.9603238
7: -0.2843126, 0.3989255, -0.6897659, 0.7465189, -1.0308315, 1.0886915
8: -0.3104765, 0.3390416, -0.6552320, 0.7644134, -1.0748899, 0.9942735
9: -0.1766882, 0.2197552, -0.5378215, 0.5958655, -0.7725537, 0.7575767

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A2_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8310343, upper bound: 1.7997559
time: 2.04 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8336240, upper bound: 1.8047420
time: 2.02 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1413150, 0.8169245, -0.1170832, 0.7466480, -0.8879629, 0.9340077
1: -0.2475138, 0.2780724, -0.2178402, 0.2473908, -0.4949046, 0.4959127
2: -0.3386415, 0.3268481, -0.3067005, 0.2921665, -0.6308080, 0.6335486
3: -0.2384097, 0.1617105, -0.2174114, 0.1282197, -0.3666295, 0.3791219
4: -0.2370225, 0.3185810, -0.2038315, 0.2947860, -0.5318085, 0.5224125
5: -0.3900245, 0.4061506, -0.3603052, 0.3766796, -0.7667041, 0.7664559
6: -0.0030257, 1.2716861, 0.0710058, 1.2627865, -1.2658122, 1.2006803
7: -0.2602233, 0.3821868, -0.2214801, 0.3545782, -0.6148015, 0.6036670
8: -0.2945766, 0.3103696, -0.2667953, 0.2749565, -0.5695331, 0.5771649
9: -0.1555537, 0.2072324, -0.1284932, 0.1896630, -0.3452168, 0.3357256

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8767039, upper bound: 1.8740969
time: 2.63 seconds

## Relational analysis of IS_A1_A2_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8767138, upper bound: 1.8738011
time: 2.22 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1243831, 0.7610237, -0.1315861, 0.7875699, -0.9119530, 0.8926098
1: -0.2252646, 0.2673297, -0.2351888, 0.2708704, -0.4961350, 0.5025185
2: -0.3149672, 0.3118170, -0.3259391, 0.3173927, -0.6323599, 0.6377561
3: -0.2234371, 0.1475389, -0.2303033, 0.1516239, -0.3750610, 0.3778423
4: -0.2139593, 0.3091208, -0.2240233, 0.3128085, -0.5267678, 0.5331441
5: -0.3648891, 0.3931568, -0.3771718, 0.3984623, -0.7633513, 0.7703286
6: 0.0501700, 1.2698017, 0.0256221, 1.2714212, -1.2212512, 1.2441796
7: -0.2429896, 0.3604816, -0.2490303, 0.3704776, -0.6134671, 0.6095119
8: -0.2812138, 0.2951046, -0.2874089, 0.3009290, -0.5821428, 0.5825135
9: -0.1467838, 0.2019943, -0.1488544, 0.2036574, -0.3504413, 0.3508487

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_A2_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8767927, upper bound: 1.8744852
time: 2.10 seconds

## Relational analysis of IS_A1_A2_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_A2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8767927, upper bound: 1.8748696
time: 2.45 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1413150, 0.8169245, -0.4267704, 1.3062426, -1.4475576, 1.2436950
1: -0.2475138, 0.2780724, -0.5163469, 0.5079370, -0.7554508, 0.7944193
2: -0.3386415, 0.3268481, -0.5893372, 0.6036969, -0.9423384, 0.9161854
3: -0.2384097, 0.1617105, -0.3909819, 0.5145843, -0.7529941, 0.5526924
4: -0.2370225, 0.3185810, -0.4949583, 0.5320820, -0.7691045, 0.8135393
5: -0.3900245, 0.4061506, -0.6386585, 0.7011141, -1.0911386, 1.0448091
6: -0.0030257, 1.2716861, -0.5658468, 1.4897968, -1.4928224, 1.8375329
7: -0.2602233, 0.3821868, -0.5776486, 0.6338608, -0.8940842, 0.9598355
8: -0.2945766, 0.3103696, -0.5443563, 0.7167417, -1.0113183, 0.8547260
9: -0.1555537, 0.2072324, -0.4393494, 0.3708146, -0.5263683, 0.6465818

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8307390, upper bound: 1.7973340
time: 2.20 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8335701, upper bound: 1.8009428
time: 1.71 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1243831, 0.7610237, -0.4553567, 1.3499527, -1.4743358, 1.2163804
1: -0.2252646, 0.2673297, -0.5419130, 0.5319115, -0.7571760, 0.8092427
2: -0.3149672, 0.3118170, -0.6124225, 0.6320112, -0.9469783, 0.9242395
3: -0.2234371, 0.1475389, -0.4050403, 0.5508766, -0.7743137, 0.5525792
4: -0.2139593, 0.3091208, -0.5192004, 0.5546509, -0.7686102, 0.8283212
5: -0.3648891, 0.3931568, -0.6630939, 0.7329614, -1.0978504, 1.0562507
6: 0.0501700, 1.2698017, -0.6166009, 1.5098497, -1.4596797, 1.8864026
7: -0.2429896, 0.3604816, -0.6097134, 0.6576377, -0.9006272, 0.9701950
8: -0.2812138, 0.2951046, -0.5670747, 0.7592266, -1.0404404, 0.8621792
9: -0.1467838, 0.2019943, -0.4687383, 0.3887703, -0.5355541, 0.6707326

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8347460, upper bound: 1.8003252
time: 2.33 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8374905, upper bound: 1.8040278
time: 2.47 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1458180, 0.8268260, -0.1549674, 0.8366895, -0.9825075, 0.9817935
1: -0.2524922, 0.2818071, -0.2599768, 0.2901302, -0.5426224, 0.5417839
2: -0.3434634, 0.3315569, -0.3497614, 0.3414122, -0.6848755, 0.6813184
3: -0.2414963, 0.1666478, -0.2453218, 0.1792192, -0.4207155, 0.4119697
4: -0.2420852, 0.3219254, -0.2489946, 0.3297234, -0.5718087, 0.5709200
5: -0.3944240, 0.4111633, -0.3988633, 0.4224702, -0.8168942, 0.8100266
6: -0.0142593, 1.2747965, -0.0297712, 1.2772847, -1.2915440, 1.3045677
7: -0.2654604, 0.3869821, -0.2766413, 0.3939434, -0.6594038, 0.6636233
8: -0.2986938, 0.3155134, -0.3067386, 0.3296145, -0.6283083, 0.6222520
9: -0.1588130, 0.2098669, -0.1687381, 0.2165681, -0.3753811, 0.3786050

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_A2_A2_B2_B1_A1_A1

### Relational analysis result of IS_A1_A2_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8623473, upper bound: 1.8667542
time: 2.63 seconds

## Relational analysis of IS_A1_A2_A2_B2_B1_A1_A2

### Relational analysis result of IS_A1_A2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8682903, upper bound: 1.8678861
time: 2.48 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1299501, 0.7738131, -0.1883273, 0.8800330, -1.0099831, 0.9621404
1: -0.2314591, 0.2721861, -0.2875983, 0.3183219, -0.5497810, 0.5597844
2: -0.3209984, 0.3179221, -0.3751941, 0.3767065, -0.6977049, 0.6931162
3: -0.2272904, 0.1539653, -0.2631949, 0.2157769, -0.4430673, 0.4171602
4: -0.2203878, 0.3133776, -0.2799078, 0.3552566, -0.5756444, 0.5932854
5: -0.3704758, 0.3995179, -0.4204234, 0.4601977, -0.8306735, 0.8199413
6: 0.0358210, 1.2713792, -0.0846785, 1.2893199, -1.2534989, 1.3560576
7: -0.2498965, 0.3662519, -0.3130104, 0.4234391, -0.6733357, 0.6792623
8: -0.2865111, 0.3017334, -0.3359196, 0.3723456, -0.6588567, 0.6376530
9: -0.1509674, 0.2054690, -0.2025242, 0.2459956, -0.3969630, 0.4079932

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_A2_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8771222, upper bound: 1.8774950
time: 2.09 seconds

## Relational analysis of IS_A1_A2_A2_B2_B1_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8771222, upper bound: 1.8777315
time: 2.70 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.04 seconds
IS_A1_A1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8834210, upper bound: 1.8763758
IS_A1_A1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8848933, upper bound: 1.8766947
IS_A1_A1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8834210, upper bound: 1.8763758
IS_A1_A1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8848933, upper bound: 1.8766947
IS_A1_A1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8694154, upper bound: 1.8636618
IS_A1_A1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8738198, upper bound: 1.8646525
IS_A1_A1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8840255, upper bound: 1.8766958
IS_A1_A1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8840255, upper bound: 1.8766958
IS_A1_A1_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8635653, upper bound: 1.8341833
IS_A1_A1_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8638894, upper bound: 1.8351324
IS_A1_A1_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8640798, upper bound: 1.8335440
IS_A1_A1_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8643846, upper bound: 1.8346781
IS_A1_A1_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8481146, upper bound: 1.8156140
IS_A1_A1_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8484632, upper bound: 1.8171526
IS_A1_A1_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8489993, upper bound: 1.8156140
IS_A1_A1_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8493331, upper bound: 1.8171503
IS_A1_A1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8846882, upper bound: 1.8745727
IS_A1_A1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8846882, upper bound: 1.8745727
IS_A1_A1_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8849655, upper bound: 1.8773454
IS_A1_A1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8849655, upper bound: 1.8773454
IS_A1_A1_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8838999, upper bound: 1.8745717
IS_A1_A1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8838999, upper bound: 1.8745717
IS_A1_A1_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8841459, upper bound: 1.8773447
IS_A1_A1_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8841459, upper bound: 1.8773447
IS_A1_A1_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8509735, upper bound: 1.8185044
IS_A1_A1_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8533099, upper bound: 1.8227900
IS_A1_A1_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8513032, upper bound: 1.8202194
IS_A1_A1_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8536409, upper bound: 1.8241040
IS_A1_A1_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8482545, upper bound: 1.8141332
IS_A1_A1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8514642, upper bound: 1.8166531
IS_A1_A1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8489977, upper bound: 1.8157347
IS_A1_A1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8522350, upper bound: 1.8181751
IS_A1_A2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8610059, upper bound: 1.8633373
IS_A1_A2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8667888, upper bound: 1.8644964
IS_A1_A2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8757816, upper bound: 1.8756270
IS_A1_A2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8765094, upper bound: 1.8765051
IS_A1_A2_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8725772, upper bound: 1.8752054
IS_A1_A2_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8735507, upper bound: 1.8760842
IS_A1_A2_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8747323, upper bound: 1.8756109
IS_A1_A2_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8756370, upper bound: 1.8765060
IS_A1_A2_A1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8310373, upper bound: 1.7985680
IS_A1_A2_A1_B2_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8335583, upper bound: 1.8027418
IS_A1_A2_A1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8307919, upper bound: 1.7985679
IS_A1_A2_A1_B2_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8332663, upper bound: 1.8027418
IS_A1_A2_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8313310, upper bound: 1.7997585
IS_A1_A2_A1_B2_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8338186, upper bound: 1.8047472
IS_A1_A2_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8310343, upper bound: 1.7997559
IS_A1_A2_A1_B2_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8336240, upper bound: 1.8047420
IS_A1_A2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8767039, upper bound: 1.8740969
IS_A1_A2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8767138, upper bound: 1.8738011
IS_A1_A2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8767927, upper bound: 1.8744852
IS_A1_A2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8767927, upper bound: 1.8748696
IS_A1_A2_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8307390, upper bound: 1.7973340
IS_A1_A2_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8335701, upper bound: 1.8009428
IS_A1_A2_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8347460, upper bound: 1.8003252
IS_A1_A2_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8374905, upper bound: 1.8040278
IS_A1_A2_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8623473, upper bound: 1.8667542
IS_A1_A2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8682903, upper bound: 1.8678861
IS_A1_A2_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8771222, upper bound: 1.8774950
IS_A1_A2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.04
Output dim: 6, lower bound: -1.8771222, upper bound: 1.8777315

## BFS IS instance: IS_A1_A1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1067069, 0.5761070, -0.1180663, 0.7542109, -0.8609177, 0.6941733
1: -0.1330298, 0.1928153, -0.2199565, 0.2571235, -0.3901534, 0.4127719
2: -0.2227838, 0.2074676, -0.3104728, 0.3006634, -0.5234472, 0.5179404
3: -0.1480890, 0.0903682, -0.2200762, 0.1348113, -0.2829003, 0.3104444
4: -0.1212737, 0.2366658, -0.2077020, 0.3008367, -0.4221104, 0.4443678
5: -0.2920136, 0.2812360, -0.3622807, 0.3802918, -0.6723053, 0.6435167
6: 0.2714193, 1.2298417, 0.0622564, 1.2598518, -0.9884325, 1.1675853
7: -0.1402428, 0.2703022, -0.2307245, 0.3569163, -0.4971592, 0.5010266
8: -0.1866002, 0.1936916, -0.2717444, 0.2823333, -0.4689334, 0.4654360
9: -0.1026372, 0.1451330, -0.1380400, 0.1930209, -0.2956581, 0.2831730

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8742755, upper bound: 1.8710898
time: 2.13 seconds

## Relational analysis of IS_A1_A1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8752268, upper bound: 1.8750765
time: 2.69 seconds

## BFS IS instance: IS_A1_A1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1086679, 0.6218267, -0.1217262, 0.7635832, -0.8722512, 0.7435529
1: -0.1625120, 0.2111322, -0.2241748, 0.2613724, -0.4238845, 0.4353070
2: -0.2497708, 0.2232981, -0.3148801, 0.3055201, -0.5552909, 0.5381782
3: -0.1729202, 0.0992132, -0.2228818, 0.1398985, -0.3128188, 0.3220950
4: -0.1456834, 0.2559192, -0.2123416, 0.3041402, -0.4498236, 0.4682609
5: -0.3121238, 0.3146966, -0.3662739, 0.3844271, -0.6965509, 0.6809704
6: 0.2068005, 1.2360606, 0.0518936, 1.2610250, -1.0542245, 1.1841670
7: -0.1650377, 0.2995016, -0.2360847, 0.3607439, -0.5257816, 0.5355864
8: -0.2117386, 0.2075311, -0.2756771, 0.2873989, -0.4991376, 0.4832082
9: -0.1051119, 0.1574605, -0.1417394, 0.1955592, -0.3006711, 0.2991999

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8753963, upper bound: 1.8713046
time: 3.06 seconds

## Relational analysis of IS_A1_A1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8763065, upper bound: 1.8752294
time: 2.74 seconds

## BFS IS instance: IS_A1_A1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1067069, 0.5761070, -0.1202419, 0.7462571, -0.8529639, 0.6963490
1: -0.1330298, 0.1928153, -0.2180424, 0.2597782, -0.3928080, 0.4108578
2: -0.2227838, 0.2074676, -0.3080242, 0.3028261, -0.5256099, 0.5154917
3: -0.1480890, 0.0903682, -0.2187989, 0.1375993, -0.2856883, 0.3091671
4: -0.1212737, 0.2366658, -0.2058253, 0.3030562, -0.4243300, 0.4424912
5: -0.2920136, 0.2812360, -0.3585544, 0.3839358, -0.6759495, 0.6397904
6: 0.2714193, 1.2298417, 0.0671780, 1.2658451, -0.9944258, 1.1626637
7: -0.1402428, 0.2703022, -0.2327982, 0.3544557, -0.4946986, 0.5031004
8: -0.1866002, 0.1936916, -0.2738879, 0.2851769, -0.4717771, 0.4675795
9: -0.1026372, 0.1451330, -0.1404051, 0.1961284, -0.2987656, 0.2855381

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_A1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8817640, upper bound: 1.8759428
time: 2.31 seconds

## Relational analysis of IS_A1_A1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8833822, upper bound: 1.8763481
time: 3.09 seconds

## BFS IS instance: IS_A1_A1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1086679, 0.6218267, -0.1226893, 0.7589805, -0.8676484, 0.7445160
1: -0.1625120, 0.2111322, -0.2239214, 0.2645709, -0.4270829, 0.4350535
2: -0.2497708, 0.2232981, -0.3138705, 0.3088518, -0.5586227, 0.5371687
3: -0.1729202, 0.0992132, -0.2224837, 0.1439407, -0.3168610, 0.3216969
4: -0.1456834, 0.2559192, -0.2122111, 0.3069932, -0.4526766, 0.4681304
5: -0.3121238, 0.3146966, -0.3641059, 0.3897264, -0.7018502, 0.6788025
6: 0.2068005, 1.2360606, 0.0530303, 1.2673900, -1.0605896, 1.1830304
7: -0.1650377, 0.2995016, -0.2396466, 0.3595882, -0.5246259, 0.5391482
8: -0.2117386, 0.2075311, -0.2788777, 0.2917179, -0.5034565, 0.4864088
9: -0.1051119, 0.1574605, -0.1443803, 0.1995707, -0.3046826, 0.3018408

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A1_B1_A1_B2_A2_A1

### Relational analysis result of IS_A1_A1_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8711023, upper bound: 1.8663616
time: 2.37 seconds

## Relational analysis of IS_A1_A1_A1_B1_A1_B2_A2_A2

### Relational analysis result of IS_A1_A1_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8749594, upper bound: 1.8673460
time: 2.63 seconds

## BFS IS instance: IS_A1_A1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1050584, 0.5024416, -0.1142360, 0.6390486, -0.7441070, 0.6166776
1: -0.1085249, 0.1643343, -0.1745063, 0.2195979, -0.3281229, 0.3388406
2: -0.1901222, 0.2076676, -0.2610435, 0.2362415, -0.4263636, 0.4687111
3: -0.1189022, 0.0763946, -0.1827321, 0.1039826, -0.2228848, 0.2591267
4: -0.0948819, 0.2086806, -0.1559063, 0.2650662, -0.3599481, 0.3645868
5: -0.2557919, 0.2449209, -0.3193130, 0.3323335, -0.5881254, 0.5642339
6: 0.3712733, 1.2281350, 0.1824799, 1.2502855, -0.8790122, 1.0456550
7: -0.1177737, 0.2243743, -0.1749392, 0.3116328, -0.4294065, 0.3993135
8: -0.1569089, 0.1932076, -0.2248056, 0.2210813, -0.3779902, 0.4180132
9: -0.0998165, 0.1293447, -0.1094620, 0.1648377, -0.2646543, 0.2388067

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_A1_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_A1_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8693995, upper bound: 1.8629515
time: 2.34 seconds

## Relational analysis of IS_A1_A1_A1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A1_A1_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8694121, upper bound: 1.8636486
time: 2.79 seconds

## BFS IS instance: IS_A1_A1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1087804, 0.6073682, -0.1163310, 0.7340022, -0.8427826, 0.7236992
1: -0.1560093, 0.2087498, -0.2130781, 0.2419354, -0.3979447, 0.4218279
2: -0.2430103, 0.2198985, -0.3013523, 0.2830361, -0.5260464, 0.5212508
3: -0.1687708, 0.0967698, -0.2133185, 0.1241049, -0.2928758, 0.3100883
4: -0.1390811, 0.2535094, -0.1978485, 0.2898242, -0.4289053, 0.4513579
5: -0.3047313, 0.3108948, -0.3562512, 0.3703595, -0.6750908, 0.6671460
6: 0.2219949, 1.2343997, 0.0847006, 1.2613269, -1.0393320, 1.1496991
7: -0.1618949, 0.2930846, -0.2137539, 0.3496556, -0.5115505, 0.5068384
8: -0.2078581, 0.2033406, -0.2605935, 0.2661352, -0.4739933, 0.4639341
9: -0.1049679, 0.1566323, -0.1217585, 0.1853436, -0.2903115, 0.2783908

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of IS_A1_A1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8738198, upper bound: 1.8646525
time: 2.93 seconds

## Relational analysis of IS_A1_A1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8738198, upper bound: 1.8646525
time: 2.72 seconds

## BFS IS instance: IS_A1_A1_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.1464125, 0.8241361, -0.1156968, 0.6870337, -0.8334461, 0.9398329
1: -0.2521383, 0.2828651, -0.1949113, 0.2335559, -0.4856942, 0.4777763
2: -0.3420772, 0.3323510, -0.2815089, 0.2694857, -0.6115630, 0.6138598
3: -0.2406743, 0.1704648, -0.2002459, 0.1149190, -0.3555933, 0.3707107
4: -0.2416758, 0.3219216, -0.1792181, 0.2808011, -0.5224769, 0.5011396
5: -0.3924929, 0.4090801, -0.3357066, 0.3562018, -0.7486947, 0.7447866
6: -0.0125664, 1.2654368, 0.1298058, 1.2539723, -1.2665386, 1.1356310
7: -0.2677125, 0.3863299, -0.2001258, 0.3317105, -0.5994231, 0.5864557
8: -0.2961976, 0.3168942, -0.2461110, 0.2516364, -0.5478340, 0.5630052
9: -0.1628464, 0.2093654, -0.1158483, 0.1791158, -0.3419623, 0.3252137

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_A1_A1_B1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8828340, upper bound: 1.8762104
time: 2.82 seconds

## Relational analysis of IS_A1_A1_A1_B1_A2_B2_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8839878, upper bound: 1.8766739
time: 2.46 seconds

## BFS IS instance: IS_A1_A1_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.1464125, 0.8241361, -0.1192355, 0.6824093, -0.8288218, 0.9433716
1: -0.2521383, 0.2828651, -0.1936613, 0.2344551, -0.4865933, 0.4765263
2: -0.3420772, 0.3323510, -0.2799360, 0.2712163, -0.6132935, 0.6122870
3: -0.2406743, 0.1704648, -0.1998027, 0.1154973, -0.3561717, 0.3702675
4: -0.2416758, 0.3219216, -0.1778986, 0.2823163, -0.5239922, 0.4998202
5: -0.3924929, 0.4090801, -0.3334808, 0.3596081, -0.7521009, 0.7425609
6: -0.0125664, 1.2654368, 0.1325760, 1.2614729, -1.2740393, 1.1328608
7: -0.2677125, 0.3863299, -0.2011686, 0.3300744, -0.5977869, 0.5874985
8: -0.2961976, 0.3168942, -0.2482372, 0.2539313, -0.5501288, 0.5651315
9: -0.1628464, 0.2093654, -0.1168694, 0.1817241, -0.3445705, 0.3262347

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_A1_A1_B1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A1_A1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8828340, upper bound: 1.8762104
time: 2.15 seconds

## Relational analysis of IS_A1_A1_A1_B1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A1_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8839878, upper bound: 1.8766739
time: 2.16 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.1520615, 0.8302066, -0.2670166, 0.9752332, -1.1272948, 1.0972232
1: -0.2566236, 0.2877554, -0.3500409, 0.3805364, -0.6371600, 0.6377962
2: -0.3460706, 0.3379225, -0.4321418, 0.4580730, -0.8041436, 0.7700642
3: -0.2429942, 0.1780821, -0.3047330, 0.2932748, -0.5362689, 0.4828151
4: -0.2459721, 0.3263655, -0.3551330, 0.4123859, -0.6583580, 0.6814985
5: -0.3956869, 0.4151320, -0.4683907, 0.5418759, -0.9375628, 0.8835227
6: -0.0210072, 1.2664195, -0.2040352, 1.3109086, -1.3319159, 1.4704547
7: -0.2740260, 0.3906524, -0.3970111, 0.4922345, -0.7662606, 0.7876635
8: -0.3003544, 0.3255997, -0.3975830, 0.4577914, -0.7581458, 0.7231827
9: -0.1692555, 0.2128653, -0.2819107, 0.3256884, -0.4949439, 0.4947760

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A1_B2_B1_B1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8497010, upper bound: 1.8179957
time: 2.42 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8519215, upper bound: 1.8222927
time: 1.91 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.1586705, 0.8426464, -0.3283972, 1.0724485, -1.2311189, 1.1710436
1: -0.2631403, 0.2928818, -0.4047883, 0.4338196, -0.6969599, 0.6976702
2: -0.3524207, 0.3442787, -0.4903904, 0.5214747, -0.8738955, 0.8346691
3: -0.2469283, 0.1858418, -0.3393748, 0.3605773, -0.6075057, 0.5252166
4: -0.2522226, 0.3312221, -0.4194081, 0.4637827, -0.7160053, 0.7506303
5: -0.4014903, 0.4219887, -0.5165806, 0.6185272, -1.0200175, 0.9385693
6: -0.0353624, 1.2701362, -0.3158168, 1.3510914, -1.3864537, 1.5859530
7: -0.2813243, 0.3968210, -0.4680108, 0.5475368, -0.8288611, 0.8648318
8: -0.3057724, 0.3346608, -0.4536993, 0.5318213, -0.8375937, 0.7883601
9: -0.1753258, 0.2165546, -0.3472822, 0.3891365, -0.5644623, 0.5638368

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A1_B2_B1_B1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8500456, upper bound: 1.8196327
time: 2.55 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8522941, upper bound: 1.8235384
time: 1.86 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.1586980, 0.8401129, -0.3752827, 1.1531340, -1.3118320, 1.2153957
1: -0.2625436, 0.2932104, -0.4482180, 0.4770408, -0.7395844, 0.7414285
2: -0.3516280, 0.3444256, -0.5381075, 0.5698073, -0.9214354, 0.8825332
3: -0.2463505, 0.1864230, -0.3663137, 0.4169253, -0.6632758, 0.5527368
4: -0.2516271, 0.3314482, -0.4705267, 0.5175871, -0.7692142, 0.8019749
5: -0.4004720, 0.4222732, -0.5556722, 0.6801582, -1.0806303, 0.9779454
6: -0.0332824, 1.2689787, -0.4085416, 1.3855797, -1.4188621, 1.6775203
7: -0.2814210, 0.3962907, -0.5272005, 0.5898204, -0.8712413, 0.9234912
8: -0.3056465, 0.3352693, -0.5037289, 0.5896147, -0.8952612, 0.8389983
9: -0.1760057, 0.2168393, -0.4003088, 0.4385968, -0.6146025, 0.6171481

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A1_B2_B1_B2_B1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8502196, upper bound: 1.8173482
time: 2.25 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B2_B1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8524225, upper bound: 1.8219172
time: 2.32 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.1657377, 0.8526633, -0.4376913, 1.2587316, -1.4244692, 1.2903546
1: -0.2691049, 0.2993187, -0.5059813, 0.5346748, -0.8037796, 0.8053000
2: -0.3578826, 0.3522834, -0.6011050, 0.6343373, -0.9922199, 0.9533885
3: -0.2509100, 0.1940204, -0.4014838, 0.4925121, -0.7434222, 0.5955042
4: -0.2591763, 0.3362103, -0.5383881, 0.5895184, -0.8486947, 0.8745984
5: -0.4063358, 0.4294157, -0.6072986, 0.7619553, -1.1682911, 1.0367142
6: -0.0478603, 1.2730515, -0.5315608, 1.4312217, -1.4790820, 1.8046123
7: -0.2889233, 0.4038035, -0.6064171, 0.6456462, -0.9345695, 1.0102205
8: -0.3120117, 0.3441406, -0.5694451, 0.6666974, -0.9787091, 0.9135857
9: -0.1830487, 0.2216592, -0.4717096, 0.5048887, -0.6879374, 0.6933688

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A1_B2_B1_B2_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8504691, upper bound: 1.8190644
time: 2.59 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B2_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8527492, upper bound: 1.8229323
time: 2.47 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.1520615, 0.8302066, -0.2820364, 0.9916130, -1.1436746, 1.1122429
1: -0.2566236, 0.2877554, -0.3619980, 0.3933043, -0.6499279, 0.6497533
2: -0.3460706, 0.3379225, -0.4428655, 0.4744057, -0.8204763, 0.7807879
3: -0.2429942, 0.1780821, -0.3124797, 0.3080887, -0.5510830, 0.4905619
4: -0.2459721, 0.3263655, -0.3688078, 0.4250236, -0.6709957, 0.6951733
5: -0.3956869, 0.4151320, -0.4779965, 0.5617892, -0.9574761, 0.8931286
6: -0.0210072, 1.2664195, -0.2261743, 1.3186002, -1.3396075, 1.4925938
7: -0.2740260, 0.3906524, -0.4131967, 0.5049489, -0.7789750, 0.8038491
8: -0.3003544, 0.3255997, -0.4125835, 0.4762360, -0.7765904, 0.7381832
9: -0.1692555, 0.2128653, -0.2958591, 0.3425085, -0.5117640, 0.5087244

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_A1_A1_B2_B2_B1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8262174, upper bound: 1.7979051
time: 2.48 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8481146, upper bound: 1.8156140
time: 2.20 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.1586705, 0.8426464, -0.3450343, 1.0978210, -1.2564914, 1.1876807
1: -0.2631403, 0.2928818, -0.4197559, 0.4498048, -0.7129450, 0.7126377
2: -0.3524207, 0.3442787, -0.5063753, 0.5394350, -0.8918557, 0.8506540
3: -0.2469283, 0.1858418, -0.3485386, 0.3805787, -0.6275070, 0.5343803
4: -0.2522226, 0.3312221, -0.4364332, 0.4838145, -0.7360371, 0.7676554
5: -0.4014903, 0.4219887, -0.5294414, 0.6436013, -1.0450916, 0.9514301
6: -0.0353624, 1.2701362, -0.3469613, 1.3612595, -1.3966218, 1.6170975
7: -0.2813243, 0.3968210, -0.4891428, 0.5619275, -0.8432518, 0.8859638
8: -0.3057724, 0.3346608, -0.4724833, 0.5539219, -0.8596943, 0.8071442
9: -0.1753258, 0.2165546, -0.3653568, 0.4082612, -0.5835870, 0.5819114

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A1_B2_B2_B1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8332376, upper bound: 1.8002001
time: 2.22 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8368414, upper bound: 1.8051321
time: 2.03 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.1586980, 0.8401129, -0.3801898, 1.1591096, -1.3178076, 1.2203027
1: -0.2625436, 0.2932104, -0.4525145, 0.4819791, -0.7445227, 0.7457250
2: -0.3516280, 0.3444256, -0.5422945, 0.5756203, -0.9272484, 0.8867202
3: -0.2463505, 0.1864230, -0.3689304, 0.4224623, -0.6688128, 0.5553534
4: -0.2516271, 0.3314482, -0.4748723, 0.5238500, -0.7754771, 0.8063205
5: -0.4004720, 0.4222732, -0.5592564, 0.6895925, -1.0900645, 0.9815297
6: -0.0332824, 1.2689787, -0.4171570, 1.3886518, -1.4219342, 1.6861358
7: -0.2814210, 0.3962907, -0.5332958, 0.5937620, -0.8751830, 0.9295865
8: -0.3056465, 0.3352693, -0.5094987, 0.5971587, -0.9028051, 0.8447680
9: -0.1760057, 0.2168393, -0.4046373, 0.4449622, -0.6209679, 0.6214766

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A1_B2_B2_B2_B1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8340248, upper bound: 1.7989241
time: 1.78 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B2_B1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8371159, upper bound: 1.8030260
time: 2.08 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.1657377, 0.8526633, -0.4422423, 1.2644372, -1.4301748, 1.2949057
1: -0.2691049, 0.2993187, -0.5099488, 0.5392246, -0.8083295, 0.8092675
2: -0.3578826, 0.3522834, -0.6050415, 0.6397091, -0.9975916, 0.9573249
3: -0.2509100, 0.1940204, -0.4039804, 0.4974778, -0.7483878, 0.5980008
4: -0.2591763, 0.3362103, -0.5423276, 0.5952454, -0.8544217, 0.8785379
5: -0.4063358, 0.4294157, -0.6107514, 0.7708721, -1.1772079, 1.0401671
6: -0.0478603, 1.2730515, -0.5396025, 1.4343398, -1.4822000, 1.8126540
7: -0.2889233, 0.4038035, -0.6119636, 0.6493099, -0.9382331, 1.0157671
8: -0.3120117, 0.3441406, -0.5745887, 0.6737241, -0.9857358, 0.9187293
9: -0.1830487, 0.2216592, -0.4754081, 0.5107316, -0.6937802, 0.6970673

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_A1_A1_B2_B2_B2_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8343434, upper bound: 1.8002001
time: 2.41 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B2_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8376282, upper bound: 1.8051321
time: 2.13 seconds

## BFS IS instance: IS_A1_A1_A2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.1142331, 0.6248693, -0.1165751, 0.6391606, -0.7533937, 0.7414445
1: -0.1684797, 0.2171838, -0.1767596, 0.2231545, -0.3916341, 0.3939434
2: -0.2546882, 0.2329551, -0.2619857, 0.2463706, -0.5010588, 0.4949407
3: -0.1785344, 0.1017102, -0.1860074, 0.1047959, -0.2833303, 0.2877175
4: -0.1497392, 0.2627160, -0.1585425, 0.2693812, -0.4191204, 0.4212584
5: -0.3125792, 0.3281232, -0.3179325, 0.3398575, -0.6524367, 0.6460558
6: 0.1964228, 1.2484608, 0.1777439, 1.2540429, -1.0576200, 1.0707169
7: -0.1719384, 0.3055758, -0.1807266, 0.3135530, -0.4854914, 0.4863024
8: -0.2208437, 0.2172101, -0.2295580, 0.2299649, -0.4508086, 0.4467680
9: -0.1093071, 0.1639370, -0.1112807, 0.1695906, -0.2788977, 0.2752177

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 54

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_A2_B1_A1_B1_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8827275, upper bound: 1.8735726
time: 1.94 seconds

## Relational analysis of IS_A1_A1_A2_B1_A1_B1_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8827275, upper bound: 1.8745727
time: 2.07 seconds

## BFS IS instance: IS_A1_A1_A2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.1142331, 0.6248693, -0.1199975, 0.6298969, -0.7441300, 0.7448668
1: -0.1684797, 0.2171838, -0.1738814, 0.2232147, -0.3916943, 0.3910652
2: -0.2546882, 0.2329551, -0.2585627, 0.2472503, -0.5019386, 0.4915177
3: -0.1785344, 0.1017102, -0.1845691, 0.1044265, -0.2829609, 0.2862793
4: -0.1497392, 0.2627160, -0.1556379, 0.2700014, -0.4197406, 0.4183539
5: -0.3125792, 0.3281232, -0.3136680, 0.3416399, -0.6542191, 0.6417912
6: 0.1964228, 1.2484608, 0.1843268, 1.2611836, -1.0647609, 1.0641340
7: -0.1719384, 0.3055758, -0.1806256, 0.3103546, -0.4822930, 0.4862014
8: -0.2208437, 0.2172101, -0.2301562, 0.2310883, -0.4519320, 0.4473662
9: -0.1093071, 0.1639370, -0.1137036, 0.1716546, -0.2809618, 0.2776406

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 54

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_A2_B1_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_A2_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8827275, upper bound: 1.8735726
time: 2.25 seconds

## Relational analysis of IS_A1_A1_A2_B1_A1_B1_B2_B2

### Relational analysis result of IS_A1_A1_A2_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8827275, upper bound: 1.8745727
time: 2.10 seconds

## BFS IS instance: IS_A1_A1_A2_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.1143816, 0.6295999, -0.1186303, 0.7241328, -0.8385144, 0.7482302
1: -0.1714844, 0.2188726, -0.2088387, 0.2492148, -0.4206992, 0.4277114
2: -0.2574211, 0.2358201, -0.2977741, 0.2910751, -0.5484962, 0.5335941
3: -0.1808843, 0.1025696, -0.2120534, 0.1270108, -0.3078951, 0.3146231
4: -0.1523274, 0.2644736, -0.1958570, 0.2945410, -0.4468683, 0.4603306
5: -0.3147168, 0.3311387, -0.3497447, 0.3736261, -0.6883429, 0.6808834
6: 0.1903963, 1.2491183, 0.0914598, 1.2616024, -1.0712061, 1.1576586
7: -0.1741680, 0.3086150, -0.2200938, 0.3459216, -0.5200896, 0.5287088
8: -0.2231580, 0.2199273, -0.2634343, 0.2731433, -0.4963013, 0.4833616
9: -0.1095108, 0.1650383, -0.1313908, 0.1899111, -0.2994218, 0.2964291

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 54

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_A2_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8831794, upper bound: 1.8756685
time: 2.39 seconds

## Relational analysis of IS_A1_A1_A2_B1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8831794, upper bound: 1.8773454
time: 2.45 seconds

## BFS IS instance: IS_A1_A1_A2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.1143816, 0.6295999, -0.1220575, 0.7191097, -0.8334913, 0.7516575
1: -0.1714844, 0.2188726, -0.2074116, 0.2508817, -0.4223661, 0.4262843
2: -0.2574211, 0.2358201, -0.2962064, 0.2924824, -0.5499035, 0.5320265
3: -0.1808843, 0.1025696, -0.2115356, 0.1279909, -0.3088751, 0.3141052
4: -0.1523274, 0.2644736, -0.1943516, 0.2962720, -0.4485993, 0.4588252
5: -0.3147168, 0.3311387, -0.3474642, 0.3766702, -0.6913871, 0.6786029
6: 0.1903963, 1.2491183, 0.0945236, 1.2686538, -1.0782574, 1.1545947
7: -0.1741680, 0.3086150, -0.2213703, 0.3441165, -0.5182844, 0.5299853
8: -0.2231580, 0.2199273, -0.2654769, 0.2751479, -0.4983059, 0.4854043
9: -0.1095108, 0.1650383, -0.1326409, 0.1923197, -0.3018305, 0.2976792

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 54

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_A2_B1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8831794, upper bound: 1.8756685
time: 2.94 seconds

## Relational analysis of IS_A1_A1_A2_B1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8831794, upper bound: 1.8773454
time: 2.75 seconds

## BFS IS instance: IS_A1_A1_A2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.1176215, 0.7494462, -0.1170430, 0.6482438, -0.7658653, 0.8664891
1: -0.2182280, 0.2551163, -0.1802633, 0.2255449, -0.4437729, 0.4353796
2: -0.3084475, 0.2986492, -0.2657512, 0.2516674, -0.5601149, 0.5644004
3: -0.2188092, 0.1326183, -0.1889545, 0.1069950, -0.3258042, 0.3215727
4: -0.2057186, 0.2996725, -0.1626331, 0.2720878, -0.4778064, 0.4623057
5: -0.3604629, 0.3799571, -0.3210957, 0.3439556, -0.7044185, 0.7010528
6: 0.0669345, 1.2618383, 0.1684060, 1.2553439, -1.1884094, 1.0934323
7: -0.2283913, 0.3552113, -0.1850235, 0.3170372, -0.5454285, 0.5402348
8: -0.2707058, 0.2807751, -0.2334319, 0.2350224, -0.5057282, 0.5142069
9: -0.1358775, 0.1927442, -0.1117316, 0.1721214, -0.3079989, 0.3044757

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_A2_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8820668, upper bound: 1.8735722
time: 7.01 seconds

## Relational analysis of IS_A1_A1_A2_B1_A2_B1_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8820668, upper bound: 1.8745717
time: 2.51 seconds

## BFS IS instance: IS_A1_A1_A2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.1176215, 0.7494462, -0.1204161, 0.6386932, -0.7563147, 0.8698623
1: -0.2182280, 0.2551163, -0.1771782, 0.2254665, -0.4436945, 0.4322945
2: -0.3084475, 0.2986492, -0.2621104, 0.2522532, -0.5607007, 0.5607596
3: -0.2188092, 0.1326183, -0.1873571, 0.1065013, -0.3253105, 0.3199754
4: -0.2057186, 0.2996725, -0.1594979, 0.2725499, -0.4782686, 0.4591705
5: -0.3604629, 0.3799571, -0.3166392, 0.3454946, -0.7059575, 0.6965964
6: 0.0669345, 1.2618383, 0.1755288, 1.2623253, -1.1953908, 1.0863096
7: -0.2283913, 0.3552113, -0.1846864, 0.3136337, -0.5420250, 0.5398977
8: -0.2707058, 0.2807751, -0.2338074, 0.2358559, -0.5065617, 0.5145825
9: -0.1358775, 0.1927442, -0.1141086, 0.1740329, -0.3099104, 0.3068527

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_A1_A2_B1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A1_A2_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8825722, upper bound: 1.8740667
time: 2.51 seconds

## Relational analysis of IS_A1_A1_A2_B1_A2_B1_B2_B2

### Relational analysis result of IS_A1_A1_A2_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8838699, upper bound: 1.8745546
time: 2.84 seconds

## BFS IS instance: IS_A1_A1_A2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.1200023, 0.7586315, -0.1191416, 0.7333314, -0.8533338, 0.8777730
1: -0.2220147, 0.2592478, -0.2123639, 0.2541724, -0.4761871, 0.4716117
2: -0.3127536, 0.3030805, -0.3020869, 0.2961776, -0.6089313, 0.6051674
3: -0.2215446, 0.1368611, -0.2149137, 0.1310268, -0.3525714, 0.3517749
4: -0.2098281, 0.3028935, -0.1999739, 0.2983293, -0.5081574, 0.5028673
5: -0.3643635, 0.3833872, -0.3534724, 0.3776017, -0.7419652, 0.7368597
6: 0.0572097, 1.2629392, 0.0817267, 1.2629716, -1.2057619, 1.1812125
7: -0.2330560, 0.3589402, -0.2254749, 0.3494868, -0.5825428, 0.5844151
8: -0.2745536, 0.2852304, -0.2677675, 0.2782226, -0.5527762, 0.5529979
9: -0.1394567, 0.1947418, -0.1356207, 0.1923591, -0.3318158, 0.3303624

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_A2_B1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8825626, upper bound: 1.8756680
time: 2.39 seconds

## Relational analysis of IS_A1_A1_A2_B1_A2_B2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8825626, upper bound: 1.8773447
time: 2.25 seconds

## BFS IS instance: IS_A1_A1_A2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.1200023, 0.7586315, -0.1225480, 0.7277572, -0.8477595, 0.8811795
1: -0.2220147, 0.2592478, -0.2108204, 0.2555345, -0.4775492, 0.4700683
2: -0.3127536, 0.3030805, -0.3002492, 0.2972766, -0.6100303, 0.6033297
3: -0.2215446, 0.1368611, -0.2142264, 0.1322380, -0.3537826, 0.3510875
4: -0.2098281, 0.3028935, -0.1982293, 0.2998290, -0.5096571, 0.5011228
5: -0.3643635, 0.3833872, -0.3509870, 0.3804063, -0.7447698, 0.7343743
6: 0.0572097, 1.2629392, 0.0853897, 1.2698758, -1.2126660, 1.1775495
7: -0.2330560, 0.3589402, -0.2264279, 0.3475079, -0.5805639, 0.5853682
8: -0.2745536, 0.2852304, -0.2695730, 0.2799161, -0.5544697, 0.5548034
9: -0.1394567, 0.1947418, -0.1366938, 0.1946172, -0.3340739, 0.3314356

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_A2_B1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8825626, upper bound: 1.8756680
time: 1.91 seconds

## Relational analysis of IS_A1_A1_A2_B1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8825626, upper bound: 1.8773447
time: 2.03 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.1158012, 0.6175687, -0.1322748, 0.7464840, -0.8622851, 0.7498436
1: -0.1674552, 0.2186076, -0.2261113, 0.2798619, -0.4473171, 0.4447189
2: -0.2524868, 0.2384254, -0.3120548, 0.3231257, -0.5756125, 0.5504801
3: -0.1793894, 0.1006459, -0.2219022, 0.1705959, -0.3499852, 0.3225481
4: -0.1490232, 0.2644373, -0.2152398, 0.3173398, -0.4663630, 0.4796771
5: -0.3081880, 0.3316331, -0.3576117, 0.4018096, -0.7099976, 0.6892447
6: 0.2000645, 1.2504803, 0.0575081, 1.2559491, -1.0558846, 1.1929722
7: -0.1738812, 0.3045976, -0.2579546, 0.3614798, -0.5353611, 0.5625521
8: -0.2220299, 0.2214960, -0.2822359, 0.3126155, -0.5346453, 0.5037318
9: -0.1104139, 0.1664108, -0.1681793, 0.2094567, -0.3198706, 0.3345901

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_B1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8484377, upper bound: 1.8173817
time: 2.47 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_B1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8484377, upper bound: 1.8185044
time: 2.90 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.1179166, 0.7178324, -0.2815170, 0.9688559, -1.0867724, 0.9993494
1: -0.2062105, 0.2454544, -0.3557638, 0.3967625, -0.6029730, 0.6012182
2: -0.2945316, 0.2871965, -0.4370426, 0.4757766, -0.7703083, 0.7242392
3: -0.2099575, 0.1242906, -0.3085897, 0.3113488, -0.5213063, 0.4328803
4: -0.1927266, 0.2915829, -0.3647083, 0.4268486, -0.6195752, 0.6562912
5: -0.3468801, 0.3704232, -0.4689819, 0.5651233, -0.9120035, 0.8394051
6: 0.0986506, 1.2597377, -0.2056101, 1.2928408, -1.1941903, 1.4653478
7: -0.2161092, 0.3432382, -0.4147198, 0.5003711, -0.7164804, 0.7579581
8: -0.2600722, 0.2691381, -0.4103912, 0.4787521, -0.7388244, 0.6795292
9: -0.1283221, 0.1879258, -0.3043883, 0.3489752, -0.4772973, 0.4923141

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_B2_A1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8516938, upper bound: 1.8214014
time: 2.25 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_B2_A2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8533096, upper bound: 1.8227900
time: 2.74 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.1159857, 0.6246842, -0.1962613, 0.8314826, -0.9474683, 0.8209456
1: -0.1712839, 0.2208448, -0.2799469, 0.3307500, -0.5020339, 0.5007917
2: -0.2560469, 0.2434613, -0.3622485, 0.3894964, -0.6455433, 0.6057098
3: -0.1823349, 0.1018445, -0.2576995, 0.2325713, -0.4149063, 0.3595440
4: -0.1534340, 0.2667657, -0.2781209, 0.3643300, -0.5177640, 0.5448866
5: -0.3114147, 0.3354436, -0.4003565, 0.4710668, -0.7824815, 0.7358000
6: 0.1914009, 1.2512522, -0.0480115, 1.2635574, -1.0721565, 1.2992637
7: -0.1769078, 0.3085161, -0.3253486, 0.4200904, -0.5969982, 0.6338648
8: -0.2250474, 0.2264314, -0.3367582, 0.3845936, -0.6096410, 0.5631896
9: -0.1106640, 0.1679358, -0.2285705, 0.2707675, -0.3814315, 0.3965063

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 54

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_B1_A1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8493814, upper bound: 1.8185589
time: 1.73 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_B1_A2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8513032, upper bound: 1.8202194
time: 2.29 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.1181138, 0.7267733, -0.3433192, 1.0724226, -1.1905365, 1.0700926
1: -0.2097337, 0.2496237, -0.4131877, 0.4525399, -0.6622735, 0.6628114
2: -0.2988213, 0.2916166, -0.4996159, 0.5393955, -0.8382167, 0.7912326
3: -0.2126500, 0.1274077, -0.3435020, 0.3849314, -0.5975814, 0.4709097
4: -0.1968146, 0.2948136, -0.4321075, 0.4882044, -0.6850190, 0.7269211
5: -0.3506771, 0.3738174, -0.5198573, 0.6452205, -0.9958976, 0.8936747
6: 0.0891067, 1.2606652, -0.3245192, 1.3307122, -1.2416055, 1.5851843
7: -0.2207032, 0.3468797, -0.4904938, 0.5563031, -0.7770064, 0.8373736
8: -0.2638337, 0.2736281, -0.4711512, 0.5548735, -0.8187072, 0.7447793
9: -0.1317143, 0.1899028, -0.3732133, 0.4139958, -0.5457101, 0.5631161

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_B2_A1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8521001, upper bound: 1.8227025
time: 2.27 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_B2_A2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8536409, upper bound: 1.8241040
time: 2.26 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1356445, 0.8119516, -0.4443254, 1.2528181, -1.3884625, 1.2562770
1: -0.2433283, 0.2703189, -0.5137379, 0.5282809, -0.7716091, 0.7840568
2: -0.3353040, 0.3183986, -0.5961043, 0.6646056, -0.9999096, 0.9145029
3: -0.2358849, 0.1517178, -0.4169280, 0.4875448, -0.7234297, 0.5686458
4: -0.2322571, 0.3120794, -0.5329449, 0.6248764, -0.8571335, 0.8450243
5: -0.3878349, 0.3950883, -0.6322988, 0.7691630, -1.1569979, 1.0273871
6: 0.0046117, 1.2676338, -0.5330508, 1.4195247, -1.4149129, 1.8006847
7: -0.2509321, 0.3786827, -0.6064990, 0.6498576, -0.9007897, 0.9851817
8: -0.2872950, 0.3009109, -0.5742098, 0.6922859, -0.9795809, 0.8751208
9: -0.1486915, 0.2004869, -0.4735731, 0.5247095, -0.6734011, 0.6740600

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_A2_B2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8466734, upper bound: 1.8140576
time: 2.66 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B1_A1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8480169, upper bound: 1.8140576
time: 2.52 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 6.45 seconds
IS_A1_A1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8742755, upper bound: 1.8710898
IS_A1_A1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8752268, upper bound: 1.8750765
IS_A1_A1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8753963, upper bound: 1.8713046
IS_A1_A1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8763065, upper bound: 1.8752294
IS_A1_A1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8817640, upper bound: 1.8759428
IS_A1_A1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8833822, upper bound: 1.8763481
IS_A1_A1_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8711023, upper bound: 1.8663616
IS_A1_A1_A1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8749594, upper bound: 1.8673460
IS_A1_A1_A1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8693995, upper bound: 1.8629515
IS_A1_A1_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8694121, upper bound: 1.8636486
IS_A1_A1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8738198, upper bound: 1.8646525
IS_A1_A1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8738198, upper bound: 1.8646525
IS_A1_A1_A1_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8828340, upper bound: 1.8762104
IS_A1_A1_A1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8839878, upper bound: 1.8766739
IS_A1_A1_A1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8828340, upper bound: 1.8762104
IS_A1_A1_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8839878, upper bound: 1.8766739
IS_A1_A1_A1_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8497010, upper bound: 1.8179957
IS_A1_A1_A1_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8519215, upper bound: 1.8222927
IS_A1_A1_A1_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8500456, upper bound: 1.8196327
IS_A1_A1_A1_B2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8522941, upper bound: 1.8235384
IS_A1_A1_A1_B2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8502196, upper bound: 1.8173482
IS_A1_A1_A1_B2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8524225, upper bound: 1.8219172
IS_A1_A1_A1_B2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8504691, upper bound: 1.8190644
IS_A1_A1_A1_B2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8527492, upper bound: 1.8229323
IS_A1_A1_A1_B2_B2_B1_B1_B1, status: Status.VERIFIED, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8262174, upper bound: 1.7979051
IS_A1_A1_A1_B2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8481146, upper bound: 1.8156140
IS_A1_A1_A1_B2_B2_B1_B2_B1, status: Status.VERIFIED, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8332376, upper bound: 1.8002001
IS_A1_A1_A1_B2_B2_B1_B2_B2, status: Status.VERIFIED, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8368414, upper bound: 1.8051321
IS_A1_A1_A1_B2_B2_B2_B1_B1, status: Status.VERIFIED, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8340248, upper bound: 1.7989241
IS_A1_A1_A1_B2_B2_B2_B1_B2, status: Status.VERIFIED, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8371159, upper bound: 1.8030260
IS_A1_A1_A1_B2_B2_B2_B2_B1, status: Status.VERIFIED, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8343434, upper bound: 1.8002001
IS_A1_A1_A1_B2_B2_B2_B2_B2, status: Status.VERIFIED, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8376282, upper bound: 1.8051321
IS_A1_A1_A2_B1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8827275, upper bound: 1.8735726
IS_A1_A1_A2_B1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8827275, upper bound: 1.8745727
IS_A1_A1_A2_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8827275, upper bound: 1.8735726
IS_A1_A1_A2_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8827275, upper bound: 1.8745727
IS_A1_A1_A2_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8831794, upper bound: 1.8756685
IS_A1_A1_A2_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8831794, upper bound: 1.8773454
IS_A1_A1_A2_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8831794, upper bound: 1.8756685
IS_A1_A1_A2_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8831794, upper bound: 1.8773454
IS_A1_A1_A2_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8820668, upper bound: 1.8735722
IS_A1_A1_A2_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8820668, upper bound: 1.8745717
IS_A1_A1_A2_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8825722, upper bound: 1.8740667
IS_A1_A1_A2_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8838699, upper bound: 1.8745546
IS_A1_A1_A2_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8825626, upper bound: 1.8756680
IS_A1_A1_A2_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8825626, upper bound: 1.8773447
IS_A1_A1_A2_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8825626, upper bound: 1.8756680
IS_A1_A1_A2_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8825626, upper bound: 1.8773447
IS_A1_A1_A2_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8484377, upper bound: 1.8173817
IS_A1_A1_A2_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8484377, upper bound: 1.8185044
IS_A1_A1_A2_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8516938, upper bound: 1.8214014
IS_A1_A1_A2_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8533096, upper bound: 1.8227900
IS_A1_A1_A2_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8493814, upper bound: 1.8185589
IS_A1_A1_A2_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8513032, upper bound: 1.8202194
IS_A1_A1_A2_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8521001, upper bound: 1.8227025
IS_A1_A1_A2_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8536409, upper bound: 1.8241040
IS_A1_A1_A2_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8466734, upper bound: 1.8140576
IS_A1_A1_A2_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.45
Output dim: 6, lower bound: -1.8480169, upper bound: 1.8140576
IS_A1_A1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8514642, upper bound: 1.8166531
IS_A1_A1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8489977, upper bound: 1.8157347
IS_A1_A1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8522350, upper bound: 1.8181751
IS_A1_A2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8610059, upper bound: 1.8633373
IS_A1_A2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8667888, upper bound: 1.8644964
IS_A1_A2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8757816, upper bound: 1.8756270
IS_A1_A2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8765094, upper bound: 1.8765051
IS_A1_A2_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8725772, upper bound: 1.8752054
IS_A1_A2_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8735507, upper bound: 1.8760842
IS_A1_A2_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8747323, upper bound: 1.8756109
IS_A1_A2_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8756370, upper bound: 1.8765060
IS_A1_A2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8767039, upper bound: 1.8740969
IS_A1_A2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8767138, upper bound: 1.8738011
IS_A1_A2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8767927, upper bound: 1.8744852
IS_A1_A2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8767927, upper bound: 1.8748696
IS_A1_A2_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8623473, upper bound: 1.8667542
IS_A1_A2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8682903, upper bound: 1.8678861
IS_A1_A2_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8771222, upper bound: 1.8774950
IS_A1_A2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.45
Output dim: 6, lower bound: -1.8771222, upper bound: 1.8777315
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.094357967376709
rel_dist={6: [-1.9113006693296792, 1.9113006693296786]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 189

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8708799, upper bound: 1.8605139
time: 2.31 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8604377, upper bound: 1.8604377
time: 3.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 5.96 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 5.96
Output dim: 6, lower bound: -1.8708799, upper bound: 1.8605139
IS_A2, status: Status.UNKNOWN, split count: 1, time: 5.96
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

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8389425, upper bound: 1.8269427
time: 3.21 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8379985, upper bound: 1.8269292
time: 2.72 seconds

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

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8325094, upper bound: 1.8268513
time: 2.30 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8268339, upper bound: 1.8268339
time: 2.93 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 6.39 seconds
IS_A1_A1, status: Status.VERIFIED, split count: 2, time: 6.39
Output dim: 6, lower bound: -1.8389425, upper bound: 1.8269427
IS_A1_A2, status: Status.VERIFIED, split count: 2, time: 6.39
Output dim: 6, lower bound: -1.8379985, upper bound: 1.8269292
IS_A2_A1, status: Status.VERIFIED, split count: 2, time: 6.39
Output dim: 6, lower bound: -1.8325094, upper bound: 1.8268513
IS_A2_A2, status: Status.VERIFIED, split count: 2, time: 6.39
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
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 189

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8824145, upper bound: 1.8626747
time: 2.25 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8625167, upper bound: 1.8625167
time: 2.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 5.13 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 5.13
Output dim: 6, lower bound: -1.8824145, upper bound: 1.8626747
IS_A2, status: Status.UNKNOWN, split count: 1, time: 5.13
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

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8524663, upper bound: 1.8288933
time: 2.41 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8505651, upper bound: 1.8288786
time: 3.06 seconds

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

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8390184, upper bound: 1.8286990
time: 1.92 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8286669, upper bound: 1.8286669
time: 2.00 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.10 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 5.10
Output dim: 6, lower bound: -1.8524663, upper bound: 1.8288933
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 5.10
Output dim: 6, lower bound: -1.8505651, upper bound: 1.8288786
IS_A2_A1, status: Status.VERIFIED, split count: 2, time: 5.10
Output dim: 6, lower bound: -1.8390184, upper bound: 1.8286990
IS_A2_A2, status: Status.VERIFIED, split count: 2, time: 5.10
Output dim: 6, lower bound: -1.8286669, upper bound: 1.8286669

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.2336920, 0.9276679, -0.3556412, 1.1344204, -1.3681124, 1.2833090
1: -0.3219900, 0.3554728, -0.4336162, 0.4565008, -0.7784908, 0.7890890
2: -0.4061271, 0.4243887, -0.5220654, 0.5489828, -0.9551098, 0.9464542
3: -0.2861868, 0.2612684, -0.3588993, 0.3871003, -0.6732871, 0.6201677
4: -0.3218213, 0.3896247, -0.4503515, 0.4910206, -0.8128418, 0.8399762
5: -0.4451328, 0.5103453, -0.5465520, 0.6565495, -1.1016823, 1.0568973
6: -0.1471975, 1.2969497, -0.3811638, 1.3973348, -1.5445323, 1.6781136
7: -0.3616745, 0.4618227, -0.4983582, 0.5756496, -0.9373241, 0.9601809
8: -0.3729174, 0.4232192, -0.4884784, 0.5653874, -0.9383048, 0.9116976
9: -0.2493206, 0.2945979, -0.3679879, 0.4136303, -0.6629509, 0.6625858

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8493603, upper bound: 1.8269673
time: 2.57 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8524304, upper bound: 1.8288933
time: 2.55 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.2360516, 0.9213381, -0.2947937, 1.0257010, -1.2617526, 1.2161318
1: -0.3221140, 0.3588542, -0.3763785, 0.4028853, -0.7249992, 0.7352327
2: -0.4051805, 0.4282180, -0.4590247, 0.4868129, -0.8919934, 0.8872427
3: -0.2861544, 0.2650071, -0.3229368, 0.3180249, -0.6041794, 0.5879440
4: -0.3222730, 0.3931868, -0.3838087, 0.4349946, -0.7572677, 0.7769955
5: -0.4428855, 0.5166113, -0.4943026, 0.5787303, -1.0216159, 1.0109138
6: -0.1440104, 1.2948549, -0.2597365, 1.3469400, -1.4909505, 1.5545914
7: -0.3651637, 0.4619819, -0.4257802, 0.5193402, -0.8845038, 0.8877621
8: -0.3766791, 0.4280382, -0.4268837, 0.4911856, -0.8678646, 0.8549218
9: -0.2533768, 0.3006897, -0.3031054, 0.3519245, -0.6053013, 0.6037951

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8463266, upper bound: 1.8263452
time: 2.07 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8505159, upper bound: 1.8288786
time: 2.57 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.97 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 5.97
Output dim: 6, lower bound: -1.8493603, upper bound: 1.8269673
IS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 5.97
Output dim: 6, lower bound: -1.8524304, upper bound: 1.8288933
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 5.97
Output dim: 6, lower bound: -1.8463266, upper bound: 1.8263452
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 5.97
Output dim: 6, lower bound: -1.8505159, upper bound: 1.8288786

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2382691, 0.9507946, -0.2956184, 1.0415601, -1.2798291, 1.2464130
1: -0.3297816, 0.3564569, -0.3805759, 0.4012115, -0.7309932, 0.7370328
2: -0.4150444, 0.4270655, -0.4639789, 0.4860690, -0.9011133, 0.8910444
3: -0.2916944, 0.2636827, -0.3259237, 0.3169423, -0.6086367, 0.5896064
4: -0.3300874, 0.3899961, -0.3875848, 0.4334022, -0.7634895, 0.7775809
5: -0.4548751, 0.5089113, -0.5004990, 0.5752178, -1.0300930, 1.0094103
6: -0.1680785, 1.3109281, -0.2729158, 1.3626550, -1.5307336, 1.5838439
7: -0.3655823, 0.4693472, -0.4255382, 0.5231442, -0.8887266, 0.8948854
8: -0.3765907, 0.4246833, -0.4323989, 0.4892783, -0.8658690, 0.8570822
9: -0.2510976, 0.2932430, -0.3003026, 0.3480596, -0.5991572, 0.5935456

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8484055, upper bound: 1.8269175
time: 2.66 seconds

## Relational analysis of IS_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8490243, upper bound: 1.8269175
time: 2.72 seconds

## BFS IS instance: IS_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.2051244, 0.8915004, -0.3458430, 1.1188074, -1.3239318, 1.2373433
1: -0.2987980, 0.3325009, -0.4247639, 0.4472773, -0.7460753, 0.7572647
2: -0.3846641, 0.3945331, -0.5124141, 0.5387520, -0.9234161, 0.9069473
3: -0.2706087, 0.2334781, -0.3535091, 0.3750729, -0.6456816, 0.5869872
4: -0.2944325, 0.3679506, -0.4398734, 0.4794792, -0.7739117, 0.8078239
5: -0.4267558, 0.4781152, -0.5388366, 0.6433467, -1.0701025, 1.0169518
6: -0.1026776, 1.2858521, -0.3626525, 1.3915666, -1.4942443, 1.6485046
7: -0.3313026, 0.4365475, -0.4858291, 0.5670593, -0.8983619, 0.9223766
8: -0.3483621, 0.3910702, -0.4784802, 0.5530831, -0.9014452, 0.8695505
9: -0.2218433, 0.2653263, -0.3565761, 0.4029009, -0.6247442, 0.6219024

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_A2_A1

### Relational analysis result of IS_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8523977, upper bound: 1.8288852
time: 3.03 seconds

## Relational analysis of IS_A1_A1_A2_A2

### Relational analysis result of IS_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8523908, upper bound: 1.8288849
time: 2.35 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2565551, 0.9744402, -0.2347029, 0.9492833, -1.2058384, 1.2091432
1: -0.3448424, 0.3715456, -0.3277945, 0.3539088, -0.6987513, 0.6993400
2: -0.4289511, 0.4466715, -0.4135317, 0.4237658, -0.8527169, 0.8602031
3: -0.3015637, 0.2813208, -0.2902621, 0.2595143, -0.5610780, 0.5715829
4: -0.3472072, 0.4048828, -0.3264608, 0.3886588, -0.7358660, 0.7313436
5: -0.4669372, 0.5324135, -0.4541011, 0.5089813, -0.9759185, 0.9865146
6: -0.1975362, 1.3229021, -0.1659564, 1.3154888, -1.5130250, 1.4888585
7: -0.3851250, 0.4853222, -0.3616607, 0.4665899, -0.8517148, 0.8469828
8: -0.3923242, 0.4465339, -0.3745522, 0.4224271, -0.8147513, 0.8210860
9: -0.2674738, 0.3125915, -0.2450077, 0.2897670, -0.5572408, 0.5575992

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8453824, upper bound: 1.8262926
time: 2.48 seconds

## Relational analysis of IS_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8461357, upper bound: 1.8262926
time: 2.76 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.2067769, 0.8844675, -0.2850382, 1.0127528, -1.2195296, 1.1695057
1: -0.2982824, 0.3353040, -0.3684385, 0.3947017, -0.6929842, 0.7037424
2: -0.3830329, 0.3976636, -0.4510797, 0.4766155, -0.8596485, 0.8487433
3: -0.2701186, 0.2365419, -0.3175932, 0.3085237, -0.5786422, 0.5541351
4: -0.2941514, 0.3709590, -0.3744408, 0.4273911, -0.7215424, 0.7453998
5: -0.4239522, 0.4835648, -0.4875089, 0.5670276, -0.9909799, 0.9710737
6: -0.0985513, 1.2864097, -0.2439280, 1.3417172, -1.4402685, 1.5303377
7: -0.3340706, 0.4360552, -0.4152108, 0.5107281, -0.8447987, 0.8512661
8: -0.3514939, 0.3950947, -0.4184920, 0.4797413, -0.8312352, 0.8135867
9: -0.2252686, 0.2706831, -0.2931941, 0.3419105, -0.5671790, 0.5638772

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8377924, upper bound: 1.8154775
time: 2.87 seconds

## Relational analysis of IS_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8385641, upper bound: 1.8170402
time: 2.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 6.84 seconds
IS_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 6, lower bound: -1.8484055, upper bound: 1.8269175
IS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 6, lower bound: -1.8490243, upper bound: 1.8269175
IS_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 6, lower bound: -1.8523977, upper bound: 1.8288852
IS_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 6, lower bound: -1.8523908, upper bound: 1.8288849
IS_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 6, lower bound: -1.8453824, upper bound: 1.8262926
IS_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 6, lower bound: -1.8461357, upper bound: 1.8262926
IS_A1_A2_A2_B1, status: Status.VERIFIED, split count: 4, time: 6.84
Output dim: 6, lower bound: -1.8377924, upper bound: 1.8154775
IS_A1_A2_A2_B2, status: Status.VERIFIED, split count: 4, time: 6.84
Output dim: 6, lower bound: -1.8385641, upper bound: 1.8170402

## BFS IS instance: IS_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2057324, 0.9096044, -0.1546529, 0.8448237, -1.0505561, 1.0642573
1: -0.3033805, 0.3303865, -0.2616872, 0.2887685, -0.5921490, 0.5920737
2: -0.3904969, 0.3931105, -0.3522544, 0.3403825, -0.7308793, 0.7453648
3: -0.2740391, 0.2320755, -0.2470719, 0.1766146, -0.4506537, 0.4791474
4: -0.2989279, 0.3654602, -0.2509662, 0.3284989, -0.6274269, 0.6164265
5: -0.4341263, 0.4723517, -0.4023592, 0.4207584, -0.8548846, 0.8747109
6: -0.1170162, 1.2946928, -0.0351552, 1.2806807, -1.3976969, 1.3298479
7: -0.3310617, 0.4405701, -0.2753910, 0.3958420, -0.7269037, 0.7159612
8: -0.3485405, 0.3882600, -0.3065475, 0.3268348, -0.6753753, 0.6948075
9: -0.2198862, 0.2601222, -0.1660362, 0.2150007, -0.4348869, 0.4261584

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8350976, upper bound: 1.8146489
time: 2.80 seconds

## Relational analysis of IS_A1_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8361802, upper bound: 1.8151254
time: 2.56 seconds

## BFS IS instance: IS_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2173750, 0.9240394, -0.2519752, 0.9859306, -1.2033055, 1.1760145
1: -0.3128095, 0.3397332, -0.3452494, 0.3661720, -0.6789815, 0.6849826
2: -0.3992631, 0.4052808, -0.4313236, 0.4406043, -0.8398674, 0.8366045
3: -0.2803435, 0.2433944, -0.3024066, 0.2745072, -0.5548508, 0.5458010
4: -0.3100467, 0.3742654, -0.3457257, 0.4004615, -0.7105082, 0.7199911
5: -0.4414645, 0.4855186, -0.4711578, 0.5264755, -0.9679401, 0.9566764
6: -0.1352555, 1.3002714, -0.2051329, 1.3385985, -1.4738541, 1.5054042
7: -0.3434257, 0.4508367, -0.3792855, 0.4846724, -0.8280981, 0.8301221
8: -0.3585448, 0.4013273, -0.3950752, 0.4404004, -0.7989452, 0.7964025
9: -0.2310466, 0.2720081, -0.2582203, 0.3034669, -0.5345135, 0.5302284

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8356345, upper bound: 1.8146489
time: 2.89 seconds

## Relational analysis of IS_A1_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8367748, upper bound: 1.8151254
time: 2.18 seconds

## BFS IS instance: IS_A1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.1159616, 0.7033735, -0.3113786, 1.0637677, -1.1797292, 1.0147521
1: -0.2010893, 0.2369420, -0.3935463, 0.4157329, -0.6168222, 0.6304883
2: -0.2882367, 0.2758111, -0.4784321, 0.5028445, -0.7910812, 0.7542433
3: -0.2049150, 0.1182289, -0.3345451, 0.3334312, -0.5383462, 0.4527740
4: -0.1859916, 0.2844449, -0.4028955, 0.4468600, -0.6328517, 0.6873404
5: -0.3422161, 0.3617626, -0.5116646, 0.5975378, -0.9397538, 0.8734273
6: 0.1144768, 1.2558804, -0.2995923, 1.3714095, -1.2569327, 1.5554726
7: -0.2056796, 0.3379471, -0.4435784, 0.5367165, -0.7423960, 0.7815256
8: -0.2516249, 0.2579985, -0.4457631, 0.5098110, -0.7614359, 0.7037616
9: -0.1192000, 0.1819619, -0.3175226, 0.3652286, -0.4844286, 0.4994845

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_A2_A1_B1

### Relational analysis result of IS_A1_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8394401, upper bound: 1.8154962
time: 2.94 seconds

## Relational analysis of IS_A1_A1_A2_A1_B2

### Relational analysis result of IS_A1_A1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8404245, upper bound: 1.8170564
time: 2.19 seconds

## BFS IS instance: IS_A1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.1622779, 0.8381748, -0.3227404, 1.0820181, -1.2442961, 1.1609151
1: -0.2638702, 0.2978253, -0.4038674, 0.4259661, -0.6898363, 0.7016927
2: -0.3524194, 0.3495883, -0.4896337, 0.5146883, -0.8671077, 0.8392220
3: -0.2470456, 0.1912882, -0.3408166, 0.3470050, -0.5940506, 0.5321048
4: -0.2530104, 0.3358105, -0.4150921, 0.4561186, -0.7091290, 0.7509025
5: -0.3999231, 0.4305325, -0.5206606, 0.6126220, -1.0125451, 0.9511930
6: -0.0352900, 1.2740173, -0.3200473, 1.3781383, -1.4134283, 1.5940647
7: -0.2856037, 0.3983221, -0.4571575, 0.5467545, -0.8323582, 0.8554797
8: -0.3113918, 0.3422747, -0.4561037, 0.5241127, -0.8355045, 0.7983783
9: -0.1802069, 0.2218284, -0.3301173, 0.3776510, -0.5578579, 0.5519457

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_A2_A2_B1

### Relational analysis result of IS_A1_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8394203, upper bound: 1.8154950
time: 2.16 seconds

## Relational analysis of IS_A1_A1_A2_A2_B2

### Relational analysis result of IS_A1_A1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8403786, upper bound: 1.8170547
time: 2.90 seconds

## BFS IS instance: IS_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2249978, 0.9339218, -0.1209015, 0.7645260, -0.9895238, 1.0548233
1: -0.3192238, 0.3462133, -0.2244809, 0.2535027, -0.5727265, 0.5706943
2: -0.4051819, 0.4137595, -0.3145765, 0.2990420, -0.7042239, 0.7283360
3: -0.2844546, 0.2506626, -0.2224760, 0.1331839, -0.4176385, 0.4731387
4: -0.3169079, 0.3810181, -0.2112300, 0.2994928, -0.6164006, 0.5922481
5: -0.4464078, 0.4970114, -0.3677163, 0.3815798, -0.8279876, 0.8647277
6: -0.1481194, 1.3069171, 0.0533588, 1.2634656, -1.4115851, 1.2535583
7: -0.3516481, 0.4573998, -0.2289443, 0.3611926, -0.7128407, 0.6863441
8: -0.3651549, 0.4111210, -0.2725902, 0.2816108, -0.6467658, 0.6837112
9: -0.2371663, 0.2803841, -0.1336207, 0.1922941, -0.4294604, 0.4140047

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_A1_B1_A1

### Relational analysis result of IS_A1_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8320351, upper bound: 1.8138988
time: 2.63 seconds

## Relational analysis of IS_A1_A2_A1_B1_A2

### Relational analysis result of IS_A1_A2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8330120, upper bound: 1.8143564
time: 2.46 seconds

## BFS IS instance: IS_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2360080, 0.9479999, -0.1911740, 0.8947601, -1.1307681, 1.1391739
1: -0.3281385, 0.3550576, -0.2925669, 0.3190471, -0.6471856, 0.6476245
2: -0.4134579, 0.4252458, -0.3808227, 0.3785057, -0.7919636, 0.8060685
3: -0.2904056, 0.2613704, -0.2667067, 0.2171860, -0.5075915, 0.5280771
4: -0.3274361, 0.3893474, -0.2848712, 0.3557812, -0.6832172, 0.6742186
5: -0.4534424, 0.5093709, -0.4265381, 0.4603044, -0.9137468, 0.9359089
6: -0.1653177, 1.3123032, -0.0979784, 1.2957108, -1.4610285, 1.4102815
7: -0.3633335, 0.4671364, -0.3154247, 0.4282633, -0.7915968, 0.7825611
8: -0.3746125, 0.4234758, -0.3372452, 0.3736328, -0.7482453, 0.7607210
9: -0.2477502, 0.2916358, -0.2031297, 0.2452857, -0.4930360, 0.4947655

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 222
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 222
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_A1_B2_A1

### Relational analysis result of IS_A1_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8327341, upper bound: 1.8138988
time: 2.84 seconds

## Relational analysis of IS_A1_A2_A1_B2_A2

### Relational analysis result of IS_A1_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8338420, upper bound: 1.8143541
time: 2.73 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 7.08 seconds
IS_A1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.08
Output dim: 6, lower bound: -1.8350976, upper bound: 1.8146489
IS_A1_A1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.08
Output dim: 6, lower bound: -1.8361802, upper bound: 1.8151254
IS_A1_A1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.08
Output dim: 6, lower bound: -1.8356345, upper bound: 1.8146489
IS_A1_A1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.08
Output dim: 6, lower bound: -1.8367748, upper bound: 1.8151254
IS_A1_A1_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 7.08
Output dim: 6, lower bound: -1.8394401, upper bound: 1.8154962
IS_A1_A1_A2_A1_B2, status: Status.VERIFIED, split count: 5, time: 7.08
Output dim: 6, lower bound: -1.8404245, upper bound: 1.8170564
IS_A1_A1_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 7.08
Output dim: 6, lower bound: -1.8394203, upper bound: 1.8154950
IS_A1_A1_A2_A2_B2, status: Status.VERIFIED, split count: 5, time: 7.08
Output dim: 6, lower bound: -1.8403786, upper bound: 1.8170547
IS_A1_A2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.08
Output dim: 6, lower bound: -1.8320351, upper bound: 1.8138988
IS_A1_A2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.08
Output dim: 6, lower bound: -1.8330120, upper bound: 1.8143564
IS_A1_A2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.08
Output dim: 6, lower bound: -1.8327341, upper bound: 1.8138988
IS_A1_A2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.08
Output dim: 6, lower bound: -1.8338420, upper bound: 1.8143541
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.094357967376709
rel_dist={6: [-1.90967747725962, 1.9096774281054198]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 1330.00 seconds
