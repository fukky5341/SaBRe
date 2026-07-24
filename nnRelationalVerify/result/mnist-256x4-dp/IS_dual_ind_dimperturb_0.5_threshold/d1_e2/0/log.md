## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0010557


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0006430, 0.0014076, 0.0006430, 0.0014076, -0.0007559, 0.0007559)
1: (0.9929122, 0.9948404, 0.9929122, 0.9948404, -0.0018197, 0.0018197)
2: (-0.0070471, -0.0046028, -0.0070471, -0.0046028, -0.0024443, 0.0024443)
3: (0.0034838, 0.0042836, 0.0034838, 0.0042836, -0.0007262, 0.0007262)
4: (0.0023745, 0.0039867, 0.0023745, 0.0039867, -0.0016122, 0.0016122)
5: (0.0052668, 0.0071408, 0.0052668, 0.0071408, -0.0018740, 0.0018740)
6: (-0.0015672, -0.0006989, -0.0015672, -0.0006989, -0.0008683, 0.0008683)
7: (-0.0087539, -0.0074045, -0.0087539, -0.0074045, -0.0013494, 0.0013494)
8: (0.0032425, 0.0075107, 0.0032425, 0.0075107, -0.0038406, 0.0038406)
9: (-0.0046863, -0.0022392, -0.0046863, -0.0022392, -0.0024472, 0.0024472)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.17 + 1.76 = 2.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0011130, upper bound: 0.0011130

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010246, upper bound: 0.0010859
time: 0.97 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010931, upper bound: 0.0010931
time: 1.04 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.12 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.12
Output dim: 1, lower bound: -0.0010246, upper bound: 0.0010859
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.12
Output dim: 1, lower bound: -0.0010931, upper bound: 0.0010931

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0006376, 0.0013295, 0.0006446, 0.0013952, -0.0007486, 0.0006759
1: 0.9931093, 0.9948539, 0.9929436, 0.9948363, -0.0016162, 0.0017989
2: -0.0067041, -0.0045900, -0.0069926, -0.0046067, -0.0020975, 0.0024026
3: 0.0034782, 0.0042018, 0.0034855, 0.0042706, -0.0007166, 0.0006416
4: 0.0023682, 0.0037156, 0.0023764, 0.0039436, -0.0015753, 0.0013392
5: 0.0052535, 0.0069492, 0.0052708, 0.0071103, -0.0018568, 0.0016783
6: -0.0014482, -0.0006942, -0.0015483, -0.0007003, -0.0007479, 0.0008541
7: -0.0086159, -0.0073949, -0.0087319, -0.0074074, -0.0012085, 0.0013370
8: 0.0032130, 0.0070600, 0.0032515, 0.0074390, -0.0037899, 0.0033926
9: -0.0044361, -0.0022219, -0.0046466, -0.0022444, -0.0021917, 0.0024247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010246, upper bound: 0.0010246
time: 1.09 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010246, upper bound: 0.0010859
time: 1.02 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0006444, 0.0013990, 0.0006430, 0.0014076, -0.0007545, 0.0007466
1: 0.9929339, 0.9948367, 0.9929122, 0.9948404, -0.0017901, 0.0018161
2: -0.0070092, -0.0046062, -0.0070471, -0.0046028, -0.0024065, 0.0024409
3: 0.0034853, 0.0042745, 0.0034838, 0.0042836, -0.0007248, 0.0007118
4: 0.0023762, 0.0039567, 0.0023745, 0.0039867, -0.0016105, 0.0015822
5: 0.0052703, 0.0071196, 0.0052668, 0.0071408, -0.0018705, 0.0018528
6: -0.0015540, -0.0007001, -0.0015672, -0.0006989, -0.0008552, 0.0008671
7: -0.0087386, -0.0074070, -0.0087539, -0.0074045, -0.0013341, 0.0013468
8: 0.0032504, 0.0074609, 0.0032425, 0.0075107, -0.0038360, 0.0036660
9: -0.0046587, -0.0022438, -0.0046863, -0.0022392, -0.0024195, 0.0024425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010859, upper bound: 0.0010246
time: 0.82 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010859, upper bound: 0.0010931
time: 0.98 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.10 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 3.10
Output dim: 1, lower bound: -0.0010246, upper bound: 0.0010246
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.10
Output dim: 1, lower bound: -0.0010246, upper bound: 0.0010859
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.10
Output dim: 1, lower bound: -0.0010859, upper bound: 0.0010246
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.10
Output dim: 1, lower bound: -0.0010859, upper bound: 0.0010931

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0006376, 0.0013295, 0.0006444, 0.0013990, -0.0007524, 0.0006761
1: 0.9931093, 0.9948539, 0.9929339, 0.9948367, -0.0016167, 0.0018089
2: -0.0067041, -0.0045900, -0.0070092, -0.0046062, -0.0020979, 0.0024193
3: 0.0034782, 0.0042018, 0.0034853, 0.0042745, -0.0007215, 0.0006418
4: 0.0023682, 0.0037156, 0.0023762, 0.0039567, -0.0015885, 0.0013394
5: 0.0052535, 0.0069492, 0.0052703, 0.0071196, -0.0018661, 0.0016788
6: -0.0014482, -0.0006942, -0.0015540, -0.0007001, -0.0007480, 0.0008599
7: -0.0086159, -0.0073949, -0.0087386, -0.0074070, -0.0012088, 0.0013437
8: 0.0032130, 0.0070600, 0.0032504, 0.0074609, -0.0038238, 0.0033933
9: -0.0044361, -0.0022219, -0.0046587, -0.0022438, -0.0021923, 0.0024368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0009988, upper bound: 0.0010583
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010025, upper bound: 0.0010645
time: 0.82 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006444, 0.0013990, 0.0006376, 0.0013295, -0.0006761, 0.0007524
1: 0.9929339, 0.9948367, 0.9931093, 0.9948539, -0.0018089, 0.0016167
2: -0.0070092, -0.0046062, -0.0067041, -0.0045900, -0.0024193, 0.0020979
3: 0.0034853, 0.0042745, 0.0034782, 0.0042018, -0.0006418, 0.0007215
4: 0.0023762, 0.0039567, 0.0023682, 0.0037156, -0.0013394, 0.0015885
5: 0.0052703, 0.0071196, 0.0052535, 0.0069492, -0.0016788, 0.0018661
6: -0.0015540, -0.0007001, -0.0014482, -0.0006942, -0.0008599, 0.0007480
7: -0.0087386, -0.0074070, -0.0086159, -0.0073949, -0.0013437, 0.0012088
8: 0.0032504, 0.0074609, 0.0032130, 0.0070600, -0.0033933, 0.0038238
9: -0.0046587, -0.0022438, -0.0044361, -0.0022219, -0.0024368, 0.0021923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010547, upper bound: 0.0009957
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010645, upper bound: 0.0010025
time: 1.14 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006444, 0.0013990, 0.0006444, 0.0013990, -0.0007452, 0.0007452
1: 0.9929339, 0.9948367, 0.9929339, 0.9948367, -0.0017865, 0.0017865
2: -0.0070092, -0.0046062, -0.0070092, -0.0046062, -0.0024030, 0.0024030
3: 0.0034853, 0.0042745, 0.0034853, 0.0042745, -0.0007103, 0.0007103
4: 0.0023762, 0.0039567, 0.0023762, 0.0039567, -0.0015805, 0.0015805
5: 0.0052703, 0.0071196, 0.0052703, 0.0071196, -0.0018493, 0.0018493
6: -0.0015540, -0.0007001, -0.0015540, -0.0007001, -0.0008539, 0.0008539
7: -0.0087386, -0.0074070, -0.0087386, -0.0074070, -0.0013316, 0.0013316
8: 0.0032504, 0.0074609, 0.0032504, 0.0074609, -0.0036609, 0.0036609
9: -0.0046587, -0.0022438, -0.0046587, -0.0022438, -0.0024149, 0.0024149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010547, upper bound: 0.0010190
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010645, upper bound: 0.0010295
time: 1.18 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.46 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 1, lower bound: -0.0009988, upper bound: 0.0010583
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 1, lower bound: -0.0010025, upper bound: 0.0010645
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.46
Output dim: 1, lower bound: -0.0010547, upper bound: 0.0009957
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 1, lower bound: -0.0010645, upper bound: 0.0010025
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.46
Output dim: 1, lower bound: -0.0010547, upper bound: 0.0010190
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 1, lower bound: -0.0010645, upper bound: 0.0010295

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0006620, 0.0013380, 0.0006478, 0.0013990, -0.0007281, 0.0006813
1: 0.9930876, 0.9947925, 0.9929341, 0.9948283, -0.0016302, 0.0017485
2: -0.0067417, -0.0046477, -0.0070090, -0.0046141, -0.0021276, 0.0023613
3: 0.0035037, 0.0042107, 0.0034888, 0.0042745, -0.0006965, 0.0006470
4: 0.0023966, 0.0037453, 0.0023801, 0.0039565, -0.0015600, 0.0013652
5: 0.0053133, 0.0069702, 0.0052785, 0.0071195, -0.0018062, 0.0016916
6: -0.0014612, -0.0007153, -0.0015540, -0.0007030, -0.0007582, 0.0008387
7: -0.0086310, -0.0074380, -0.0087385, -0.0074130, -0.0012181, 0.0013005
8: 0.0033461, 0.0071094, 0.0032687, 0.0074606, -0.0036485, 0.0033562
9: -0.0044635, -0.0022999, -0.0046585, -0.0022545, -0.0022090, 0.0023586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0009475, upper bound: 0.0010178
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0009829, upper bound: 0.0010432
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006521, 0.0013294, 0.0006444, 0.0013990, -0.0007380, 0.0006760
1: 0.9931094, 0.9948172, 0.9929339, 0.9948367, -0.0016166, 0.0017751
2: -0.0067040, -0.0046245, -0.0070092, -0.0046062, -0.0020978, 0.0023848
3: 0.0034934, 0.0042017, 0.0034853, 0.0042745, -0.0007087, 0.0006418
4: 0.0023852, 0.0037155, 0.0023762, 0.0039567, -0.0015716, 0.0013393
5: 0.0052893, 0.0069491, 0.0052703, 0.0071196, -0.0018304, 0.0016787
6: -0.0014481, -0.0007068, -0.0015540, -0.0007001, -0.0007480, 0.0008472
7: -0.0086158, -0.0074207, -0.0087386, -0.0074070, -0.0012088, 0.0013180
8: 0.0032925, 0.0070598, 0.0032504, 0.0074609, -0.0038070, 0.0033409
9: -0.0044360, -0.0022685, -0.0046587, -0.0022438, -0.0021922, 0.0023902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0009957, upper bound: 0.0010547
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0009957, upper bound: 0.0010644
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0006592, 0.0013988, 0.0006376, 0.0013295, -0.0006613, 0.0007522
1: 0.9929344, 0.9947997, 0.9931093, 0.9948539, -0.0018085, 0.0015821
2: -0.0070085, -0.0046411, -0.0067041, -0.0045900, -0.0024185, 0.0020630
3: 0.0035008, 0.0042743, 0.0034782, 0.0042018, -0.0006281, 0.0007213
4: 0.0023933, 0.0039561, 0.0023682, 0.0037156, -0.0013223, 0.0015879
5: 0.0053065, 0.0071192, 0.0052535, 0.0069492, -0.0016427, 0.0018657
6: -0.0015538, -0.0007129, -0.0014482, -0.0006942, -0.0008596, 0.0007353
7: -0.0087383, -0.0074331, -0.0086159, -0.0073949, -0.0013434, 0.0011828
8: 0.0033309, 0.0074599, 0.0032130, 0.0070600, -0.0033584, 0.0037795
9: -0.0046582, -0.0022910, -0.0044361, -0.0022219, -0.0024363, 0.0021451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010583, upper bound: 0.0009988
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010583, upper bound: 0.0010025
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006592, 0.0013988, 0.0006444, 0.0013990, -0.0007305, 0.0007450
1: 0.9929344, 0.9947997, 0.9929339, 0.9948367, -0.0017861, 0.0017520
2: -0.0070085, -0.0046411, -0.0070092, -0.0046062, -0.0024023, 0.0023681
3: 0.0035008, 0.0042743, 0.0034853, 0.0042745, -0.0006972, 0.0007101
4: 0.0023933, 0.0039561, 0.0023762, 0.0039567, -0.0015634, 0.0015799
5: 0.0053065, 0.0071192, 0.0052703, 0.0071196, -0.0018131, 0.0018489
6: -0.0015538, -0.0007129, -0.0015540, -0.0007001, -0.0008537, 0.0008412
7: -0.0087383, -0.0074331, -0.0087386, -0.0074070, -0.0013313, 0.0013056
8: 0.0033309, 0.0074599, 0.0032504, 0.0074609, -0.0036391, 0.0036121
9: -0.0046582, -0.0022910, -0.0046587, -0.0022438, -0.0024144, 0.0023677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010646, upper bound: 0.0010213
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010646, upper bound: 0.0010295
time: 1.05 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.02 seconds
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 1, lower bound: -0.0009475, upper bound: 0.0010178
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 1, lower bound: -0.0009829, upper bound: 0.0010432
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 1, lower bound: -0.0009957, upper bound: 0.0010547
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 1, lower bound: -0.0009957, upper bound: 0.0010644
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 1, lower bound: -0.0010583, upper bound: 0.0009988
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 1, lower bound: -0.0010583, upper bound: 0.0010025
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 1, lower bound: -0.0010646, upper bound: 0.0010213
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 1, lower bound: -0.0010646, upper bound: 0.0010295

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006521, 0.0013294, 0.0006592, 0.0013988, -0.0007378, 0.0006612
1: 0.9931094, 0.9948172, 0.9929344, 0.9947997, -0.0015820, 0.0017747
2: -0.0067040, -0.0046245, -0.0070085, -0.0046411, -0.0020629, 0.0023840
3: 0.0034934, 0.0042017, 0.0035008, 0.0042743, -0.0007086, 0.0006280
4: 0.0023852, 0.0037155, 0.0023933, 0.0039561, -0.0015710, 0.0013222
5: 0.0052893, 0.0069491, 0.0053065, 0.0071192, -0.0018300, 0.0016426
6: -0.0014481, -0.0007068, -0.0015538, -0.0007129, -0.0007352, 0.0008470
7: -0.0086158, -0.0074207, -0.0087383, -0.0074331, -0.0011828, 0.0013177
8: 0.0032925, 0.0070598, 0.0033309, 0.0074599, -0.0037774, 0.0033246
9: -0.0044360, -0.0022685, -0.0046582, -0.0022910, -0.0021450, 0.0023897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0009518, upper bound: 0.0010049
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0009806, upper bound: 0.0010475
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006592, 0.0013988, 0.0006620, 0.0013380, -0.0006697, 0.0007280
1: 0.9929344, 0.9947997, 0.9930876, 0.9947925, -0.0017483, 0.0016010
2: -0.0070085, -0.0046411, -0.0067417, -0.0046477, -0.0023608, 0.0021006
3: 0.0035008, 0.0042743, 0.0035037, 0.0042107, -0.0006350, 0.0006963
4: 0.0023933, 0.0039561, 0.0023966, 0.0037453, -0.0013520, 0.0015596
5: 0.0053065, 0.0071192, 0.0053133, 0.0069702, -0.0016637, 0.0018059
6: -0.0015538, -0.0007129, -0.0014612, -0.0007153, -0.0008385, 0.0007483
7: -0.0087383, -0.0074331, -0.0086310, -0.0074380, -0.0013003, 0.0011979
8: 0.0033309, 0.0074599, 0.0033461, 0.0071094, -0.0032657, 0.0036161
9: -0.0046582, -0.0022910, -0.0044635, -0.0022999, -0.0023582, 0.0021725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010178, upper bound: 0.0009475
time: 1.10 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010428, upper bound: 0.0009829
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006592, 0.0013988, 0.0006521, 0.0013294, -0.0006612, 0.0007378
1: 0.9929344, 0.9947997, 0.9931094, 0.9948172, -0.0017747, 0.0015820
2: -0.0070085, -0.0046411, -0.0067040, -0.0046245, -0.0023840, 0.0020629
3: 0.0035008, 0.0042743, 0.0034934, 0.0042017, -0.0006280, 0.0007086
4: 0.0023933, 0.0039561, 0.0023852, 0.0037155, -0.0013222, 0.0015710
5: 0.0053065, 0.0071192, 0.0052893, 0.0069491, -0.0016426, 0.0018300
6: -0.0015538, -0.0007129, -0.0014481, -0.0007068, -0.0008470, 0.0007352
7: -0.0087383, -0.0074331, -0.0086158, -0.0074207, -0.0013177, 0.0011828
8: 0.0033309, 0.0074599, 0.0032925, 0.0070598, -0.0033246, 0.0037774
9: -0.0046582, -0.0022910, -0.0044360, -0.0022685, -0.0023897, 0.0021450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010178, upper bound: 0.0009522
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010428, upper bound: 0.0009864
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006592, 0.0013988, 0.0006686, 0.0014079, -0.0007392, 0.0007209
1: 0.9929344, 0.9947997, 0.9929115, 0.9947757, -0.0017255, 0.0017719
2: -0.0070085, -0.0046411, -0.0070480, -0.0046634, -0.0023451, 0.0024069
3: 0.0035008, 0.0042743, 0.0035106, 0.0042838, -0.0007045, 0.0006852
4: 0.0023933, 0.0039561, 0.0024043, 0.0039874, -0.0015940, 0.0015519
5: 0.0053065, 0.0071192, 0.0053296, 0.0071413, -0.0018348, 0.0017896
6: -0.0015538, -0.0007129, -0.0015675, -0.0007210, -0.0008328, 0.0008546
7: -0.0087383, -0.0074331, -0.0087542, -0.0074497, -0.0012886, 0.0013212
8: 0.0033309, 0.0074599, 0.0033823, 0.0075118, -0.0035651, 0.0034530
9: -0.0046582, -0.0022910, -0.0046870, -0.0023212, -0.0023370, 0.0023960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010251, upper bound: 0.0009828
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010495, upper bound: 0.0010055
time: 1.51 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006592, 0.0013988, 0.0006592, 0.0013988, -0.0007303, 0.0007303
1: 0.9929344, 0.9947997, 0.9929344, 0.9947997, -0.0017515, 0.0017515
2: -0.0070085, -0.0046411, -0.0070085, -0.0046411, -0.0023674, 0.0023674
3: 0.0035008, 0.0042743, 0.0035008, 0.0042743, -0.0006970, 0.0006970
4: 0.0023933, 0.0039561, 0.0023933, 0.0039561, -0.0015628, 0.0015628
5: 0.0053065, 0.0071192, 0.0053065, 0.0071192, -0.0018127, 0.0018127
6: -0.0015538, -0.0007129, -0.0015538, -0.0007129, -0.0008409, 0.0008409
7: -0.0087383, -0.0074331, -0.0087383, -0.0074331, -0.0013053, 0.0013053
8: 0.0033309, 0.0074599, 0.0033309, 0.0074599, -0.0036071, 0.0036071
9: -0.0046582, -0.0022910, -0.0046582, -0.0022910, -0.0023672, 0.0023672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010251, upper bound: 0.0009948
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010495, upper bound: 0.0010129
time: 1.48 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.81 seconds
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.81
Output dim: 1, lower bound: -0.0009518, upper bound: 0.0010049
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.81
Output dim: 1, lower bound: -0.0009806, upper bound: 0.0010475
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.81
Output dim: 1, lower bound: -0.0010178, upper bound: 0.0009475
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.81
Output dim: 1, lower bound: -0.0010428, upper bound: 0.0009829
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.81
Output dim: 1, lower bound: -0.0010178, upper bound: 0.0009522
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.81
Output dim: 1, lower bound: -0.0010428, upper bound: 0.0009864
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.81
Output dim: 1, lower bound: -0.0010251, upper bound: 0.0009828
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.81
Output dim: 1, lower bound: -0.0010495, upper bound: 0.0010055
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.81
Output dim: 1, lower bound: -0.0010251, upper bound: 0.0009948
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.81
Output dim: 1, lower bound: -0.0010495, upper bound: 0.0010129

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.93 + 47.83 = 50.76 seconds
