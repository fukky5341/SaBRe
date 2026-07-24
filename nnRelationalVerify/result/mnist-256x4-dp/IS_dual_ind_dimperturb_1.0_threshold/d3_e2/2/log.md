## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00199528


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000766, 0.0000766)
1: (-0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0028695, 0.0028695)
2: (0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0034436, 0.0034436)
3: (-0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0253993, 0.0253993)
4: (-0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0019318, 0.0019318)
5: (0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0019524, 0.0019524)
6: (0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0009496, 0.0009496)
7: (-0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0065824, 0.0065824)
8: (0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0052222, 0.0052222)
9: (0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0093926, 0.0093926)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.49 + 2.42 = 3.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0024941, upper bound: 0.0024941

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0023826, upper bound: 0.0023529
time: 1.30 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0023826, upper bound: 0.0023826
time: 1.17 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.61 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.61
Output dim: 2, lower bound: -0.0023826, upper bound: 0.0023529
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.61
Output dim: 2, lower bound: -0.0023826, upper bound: 0.0023826

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0041778, -0.0040996, -0.0041779, -0.0040944, -0.0000728, 0.0000694
1: -0.0091134, -0.0061877, -0.0091192, -0.0059920, -0.0027266, 0.0026002
2: 0.9655270, 0.9690380, 0.9655201, 0.9692727, -0.0032721, 0.0031203
3: -0.0079610, 0.0179349, -0.0080124, 0.0196669, -0.0241342, 0.0230150
4: -0.0020571, -0.0000876, -0.0021888, -0.0000836, -0.0017504, 0.0018355
5: 0.0151913, 0.0171819, 0.0150582, 0.0171858, -0.0017691, 0.0018551
6: 0.0035495, 0.0045177, 0.0035476, 0.0045825, -0.0009023, 0.0008605
7: -0.0124262, -0.0057151, -0.0128751, -0.0057018, -0.0059645, 0.0062546
8: 0.0068708, 0.0121951, 0.0065146, 0.0122056, -0.0047320, 0.0049621
9: 0.0100823, 0.0196586, 0.0094418, 0.0196776, -0.0085109, 0.0089248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0022417, upper bound: 0.0021177
time: 1.14 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0022081, upper bound: 0.0021645
time: 1.20 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0041778, -0.0040962, -0.0041780, -0.0040934, -0.0000764, 0.0000693
1: -0.0091157, -0.0060596, -0.0091215, -0.0059531, -0.0028605, 0.0025960
2: 0.9655242, 0.9691916, 0.9655173, 0.9693196, -0.0034327, 0.0031153
3: -0.0079819, 0.0190687, -0.0080332, 0.0200121, -0.0253189, 0.0229781
4: -0.0021433, -0.0000860, -0.0022151, -0.0000821, -0.0017476, 0.0019257
5: 0.0151041, 0.0171835, 0.0150316, 0.0171874, -0.0017663, 0.0019462
6: 0.0035488, 0.0045601, 0.0035468, 0.0045954, -0.0009466, 0.0008591
7: -0.0127201, -0.0057097, -0.0129646, -0.0056964, -0.0059550, 0.0065616
8: 0.0066376, 0.0121994, 0.0064437, 0.0122099, -0.0047244, 0.0052057
9: 0.0096630, 0.0196663, 0.0093142, 0.0196853, -0.0084973, 0.0093629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0023521, upper bound: 0.0023826
time: 1.29 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0023521, upper bound: 0.0023826
time: 1.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.04 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.04
Output dim: 2, lower bound: -0.0022417, upper bound: 0.0021177
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.04
Output dim: 2, lower bound: -0.0022081, upper bound: 0.0021645
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.04
Output dim: 2, lower bound: -0.0023521, upper bound: 0.0023826
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.04
Output dim: 2, lower bound: -0.0023521, upper bound: 0.0023826

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041778, -0.0040996, -0.0041750, -0.0040946, -0.0000726, 0.0000667
1: -0.0091134, -0.0061877, -0.0090094, -0.0059987, -0.0027197, 0.0024987
2: 0.9655270, 0.9690380, 0.9656518, 0.9692647, -0.0032638, 0.0029985
3: -0.0079610, 0.0179349, -0.0070406, 0.0196077, -0.0240732, 0.0221167
4: -0.0020571, -0.0000876, -0.0021843, -0.0001576, -0.0016821, 0.0018309
5: 0.0151913, 0.0171819, 0.0150627, 0.0171111, -0.0017001, 0.0018505
6: 0.0035495, 0.0045177, 0.0035839, 0.0045803, -0.0009001, 0.0008269
7: -0.0124262, -0.0057151, -0.0128597, -0.0059536, -0.0057317, 0.0062388
8: 0.0068708, 0.0121951, 0.0065268, 0.0120058, -0.0045473, 0.0049496
9: 0.0100823, 0.0196586, 0.0094637, 0.0193183, -0.0081787, 0.0089023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021503, upper bound: 0.0021177
time: 1.40 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021503, upper bound: 0.0021177
time: 1.78 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041768, -0.0040997, -0.0041727, -0.0040829, -0.0000842, 0.0000700
1: -0.0090774, -0.0061901, -0.0089221, -0.0055597, -0.0031534, 0.0026219
2: 0.9655701, 0.9690351, 0.9657566, 0.9697917, -0.0037842, 0.0031464
3: -0.0076429, 0.0179142, -0.0062678, 0.0234936, -0.0279114, 0.0232074
4: -0.0020555, -0.0001117, -0.0024799, -0.0002163, -0.0017651, 0.0021228
5: 0.0151929, 0.0171574, 0.0147640, 0.0170517, -0.0017839, 0.0021455
6: 0.0035614, 0.0045170, 0.0036128, 0.0047256, -0.0010436, 0.0008677
7: -0.0124209, -0.0057975, -0.0138668, -0.0061539, -0.0060144, 0.0072335
8: 0.0068750, 0.0121297, 0.0057279, 0.0118469, -0.0047715, 0.0057387
9: 0.0100900, 0.0195410, 0.0080267, 0.0190325, -0.0085821, 0.0103216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018434, upper bound: 0.0018984
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018178, upper bound: 0.0017615
time: 1.02 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041778, -0.0040962, -0.0041778, -0.0040996, -0.0000693, 0.0000721
1: -0.0091157, -0.0060596, -0.0091134, -0.0061877, -0.0025953, 0.0026985
2: 0.9655242, 0.9691916, 0.9655270, 0.9690380, -0.0031144, 0.0032384
3: -0.0079819, 0.0190687, -0.0079610, 0.0179349, -0.0229714, 0.0238857
4: -0.0021433, -0.0000860, -0.0020571, -0.0000876, -0.0018166, 0.0017471
5: 0.0151041, 0.0171835, 0.0151913, 0.0171819, -0.0018360, 0.0017658
6: 0.0035488, 0.0045601, 0.0035495, 0.0045177, -0.0008589, 0.0008930
7: -0.0127201, -0.0057097, -0.0124262, -0.0057151, -0.0061902, 0.0059532
8: 0.0066376, 0.0121994, 0.0068708, 0.0121951, -0.0049110, 0.0047230
9: 0.0096630, 0.0196663, 0.0100823, 0.0196586, -0.0088329, 0.0084948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021176, upper bound: 0.0022417
time: 1.58 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021645, upper bound: 0.0022081
time: 1.15 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041778, -0.0040962, -0.0041778, -0.0040962, -0.0000691, 0.0000691
1: -0.0091157, -0.0060596, -0.0091157, -0.0060596, -0.0025889, 0.0025889
2: 0.9655242, 0.9691916, 0.9655242, 0.9691916, -0.0031068, 0.0031068
3: -0.0079819, 0.0190687, -0.0079819, 0.0190687, -0.0229149, 0.0229149
4: -0.0021433, -0.0000860, -0.0021433, -0.0000860, -0.0017428, 0.0017428
5: 0.0151041, 0.0171835, 0.0151041, 0.0171835, -0.0017614, 0.0017614
6: 0.0035488, 0.0045601, 0.0035488, 0.0045601, -0.0008568, 0.0008568
7: -0.0127201, -0.0057097, -0.0127201, -0.0057097, -0.0059386, 0.0059386
8: 0.0066376, 0.0121994, 0.0066376, 0.0121994, -0.0047114, 0.0047114
9: 0.0096630, 0.0196663, 0.0096630, 0.0196663, -0.0084739, 0.0084739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021176, upper bound: 0.0022416
time: 1.76 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021645, upper bound: 0.0022088
time: 1.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.88 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.88
Output dim: 2, lower bound: -0.0021503, upper bound: 0.0021177
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.88
Output dim: 2, lower bound: -0.0021503, upper bound: 0.0021177
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.88
Output dim: 2, lower bound: -0.0018434, upper bound: 0.0018984
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.88
Output dim: 2, lower bound: -0.0018178, upper bound: 0.0017615
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.88
Output dim: 2, lower bound: -0.0021176, upper bound: 0.0022417
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.88
Output dim: 2, lower bound: -0.0021645, upper bound: 0.0022081
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.88
Output dim: 2, lower bound: -0.0021176, upper bound: 0.0022416
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.88
Output dim: 2, lower bound: -0.0021645, upper bound: 0.0022088

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041749, -0.0040998, -0.0041750, -0.0040946, -0.0000697, 0.0000665
1: -0.0090040, -0.0061942, -0.0090094, -0.0059987, -0.0026099, 0.0024919
2: 0.9656582, 0.9690302, 0.9656518, 0.9692647, -0.0031320, 0.0029904
3: -0.0069931, 0.0178778, -0.0070406, 0.0196077, -0.0231013, 0.0220567
4: -0.0020527, -0.0001612, -0.0021843, -0.0001576, -0.0016775, 0.0017570
5: 0.0151957, 0.0171075, 0.0150627, 0.0171111, -0.0016954, 0.0017757
6: 0.0035857, 0.0045156, 0.0035839, 0.0045803, -0.0008637, 0.0008247
7: -0.0124114, -0.0059659, -0.0128597, -0.0059536, -0.0057162, 0.0059869
8: 0.0068825, 0.0119961, 0.0065268, 0.0120058, -0.0045349, 0.0047497
9: 0.0101035, 0.0193007, 0.0094637, 0.0193183, -0.0081565, 0.0085428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021903, upper bound: 0.0021177
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021903, upper bound: 0.0021177
time: 1.70 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041725, -0.0040883, -0.0041750, -0.0040946, -0.0000690, 0.0000789
1: -0.0089153, -0.0057622, -0.0090094, -0.0059987, -0.0025830, 0.0029547
2: 0.9657646, 0.9695485, 0.9656518, 0.9692647, -0.0030997, 0.0035458
3: -0.0062079, 0.0217011, -0.0070406, 0.0196077, -0.0228625, 0.0261531
4: -0.0023435, -0.0002209, -0.0021843, -0.0001576, -0.0019891, 0.0017388
5: 0.0149018, 0.0170471, 0.0150627, 0.0171111, -0.0020103, 0.0017574
6: 0.0036151, 0.0046586, 0.0035839, 0.0045803, -0.0008548, 0.0009778
7: -0.0134023, -0.0061694, -0.0128597, -0.0059536, -0.0067778, 0.0059250
8: 0.0060964, 0.0118346, 0.0065268, 0.0120058, -0.0053772, 0.0047006
9: 0.0086896, 0.0190103, 0.0094637, 0.0193183, -0.0096714, 0.0084545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021903, upper bound: 0.0021177
time: 1.69 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021903, upper bound: 0.0021177
time: 1.74 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041749, -0.0040964, -0.0041778, -0.0040996, -0.0000666, 0.0000719
1: -0.0090060, -0.0060662, -0.0091134, -0.0061877, -0.0024939, 0.0026922
2: 0.9656559, 0.9691837, 0.9655270, 0.9690380, -0.0029928, 0.0032308
3: -0.0070101, 0.0190106, -0.0079610, 0.0179349, -0.0220742, 0.0238299
4: -0.0021389, -0.0001599, -0.0020571, -0.0000876, -0.0018124, 0.0016789
5: 0.0151086, 0.0171088, 0.0151913, 0.0171819, -0.0018318, 0.0016968
6: 0.0035851, 0.0045580, 0.0035495, 0.0045177, -0.0008253, 0.0008910
7: -0.0127050, -0.0059615, -0.0124262, -0.0057151, -0.0061757, 0.0057207
8: 0.0066496, 0.0119996, 0.0068708, 0.0121951, -0.0048995, 0.0045386
9: 0.0096846, 0.0193070, 0.0100823, 0.0196586, -0.0088123, 0.0081630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021177, upper bound: 0.0021504
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021177, upper bound: 0.0022081
time: 1.59 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041726, -0.0040848, -0.0041768, -0.0040997, -0.0000699, 0.0000831
1: -0.0089186, -0.0056334, -0.0090774, -0.0061901, -0.0026183, 0.0031126
2: 0.9657608, 0.9697031, 0.9655701, 0.9690351, -0.0031421, 0.0037353
3: -0.0062365, 0.0228413, -0.0076429, 0.0179142, -0.0231757, 0.0275506
4: -0.0024302, -0.0002187, -0.0020555, -0.0001117, -0.0020954, 0.0017627
5: 0.0148141, 0.0170493, 0.0151929, 0.0171574, -0.0021178, 0.0017815
6: 0.0036140, 0.0047012, 0.0035614, 0.0045170, -0.0008665, 0.0010301
7: -0.0136978, -0.0061620, -0.0124209, -0.0057975, -0.0071400, 0.0060062
8: 0.0058620, 0.0118405, 0.0068750, 0.0121297, -0.0056645, 0.0047650
9: 0.0082679, 0.0190209, 0.0100900, 0.0195410, -0.0101882, 0.0085704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018984, upper bound: 0.0018434
time: 1.06 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017615, upper bound: 0.0018178
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041749, -0.0040964, -0.0041778, -0.0040962, -0.0000662, 0.0000689
1: -0.0090060, -0.0060662, -0.0091157, -0.0060596, -0.0024789, 0.0025814
2: 0.9656559, 0.9691837, 0.9655242, 0.9691916, -0.0029748, 0.0030978
3: -0.0070101, 0.0190106, -0.0079819, 0.0190687, -0.0219416, 0.0228488
4: -0.0021389, -0.0001599, -0.0021433, -0.0000860, -0.0017378, 0.0016688
5: 0.0151086, 0.0171088, 0.0151041, 0.0171835, -0.0017563, 0.0016866
6: 0.0035851, 0.0045580, 0.0035488, 0.0045601, -0.0008204, 0.0008543
7: -0.0127050, -0.0059615, -0.0127201, -0.0057097, -0.0059215, 0.0056864
8: 0.0066496, 0.0119996, 0.0066376, 0.0121994, -0.0046978, 0.0045113
9: 0.0096846, 0.0193070, 0.0096630, 0.0196663, -0.0084495, 0.0081140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021194, upper bound: 0.0021504
time: 1.26 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021194, upper bound: 0.0022088
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041726, -0.0040848, -0.0041769, -0.0040963, -0.0000694, 0.0000805
1: -0.0089186, -0.0056334, -0.0090797, -0.0060622, -0.0026006, 0.0030149
2: 0.9657608, 0.9697031, 0.9655675, 0.9691886, -0.0031208, 0.0036180
3: -0.0062365, 0.0228413, -0.0076626, 0.0190462, -0.0230187, 0.0266855
4: -0.0024302, -0.0002187, -0.0021416, -0.0001102, -0.0020296, 0.0017507
5: 0.0148141, 0.0170493, 0.0151059, 0.0171589, -0.0020513, 0.0017694
6: 0.0036140, 0.0047012, 0.0035607, 0.0045593, -0.0008606, 0.0009977
7: -0.0136978, -0.0061620, -0.0127143, -0.0057924, -0.0069158, 0.0059655
8: 0.0058620, 0.0118405, 0.0066423, 0.0121337, -0.0054866, 0.0047328
9: 0.0082679, 0.0190209, 0.0096714, 0.0195483, -0.0098683, 0.0085123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021729, upper bound: 0.0021504
time: 1.22 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021729, upper bound: 0.0022088
time: 1.24 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.94 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 2, lower bound: -0.0021903, upper bound: 0.0021177
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 2, lower bound: -0.0021903, upper bound: 0.0021177
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 2, lower bound: -0.0021903, upper bound: 0.0021177
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 2, lower bound: -0.0021903, upper bound: 0.0021177
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 2, lower bound: -0.0021177, upper bound: 0.0021504
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 2, lower bound: -0.0021177, upper bound: 0.0022081
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.94
Output dim: 2, lower bound: -0.0018984, upper bound: 0.0018434
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.94
Output dim: 2, lower bound: -0.0017615, upper bound: 0.0018178
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 2, lower bound: -0.0021194, upper bound: 0.0021504
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 2, lower bound: -0.0021194, upper bound: 0.0022088
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 2, lower bound: -0.0021729, upper bound: 0.0021504
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 2, lower bound: -0.0021729, upper bound: 0.0022088

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041749, -0.0040998, -0.0041749, -0.0040998, -0.0000639, 0.0000639
1: -0.0090040, -0.0061942, -0.0090040, -0.0061942, -0.0023944, 0.0023944
2: 0.9656582, 0.9690302, 0.9656582, 0.9690302, -0.0028734, 0.0028734
3: -0.0069931, 0.0178778, -0.0069931, 0.0178778, -0.0211938, 0.0211938
4: -0.0020527, -0.0001612, -0.0020527, -0.0001612, -0.0016119, 0.0016119
5: 0.0151957, 0.0171075, 0.0151957, 0.0171075, -0.0016291, 0.0016291
6: 0.0035857, 0.0045156, 0.0035857, 0.0045156, -0.0007924, 0.0007924
7: -0.0124114, -0.0059659, -0.0124114, -0.0059659, -0.0054926, 0.0054926
8: 0.0068825, 0.0119961, 0.0068825, 0.0119961, -0.0043575, 0.0043575
9: 0.0101035, 0.0193007, 0.0101035, 0.0193007, -0.0078375, 0.0078375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019798, upper bound: 0.0020689
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020012, upper bound: 0.0020013
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041749, -0.0040998, -0.0041749, -0.0040964, -0.0000690, 0.0000664
1: -0.0090040, -0.0061942, -0.0090060, -0.0060662, -0.0025824, 0.0024871
2: 0.9656582, 0.9690302, 0.9656559, 0.9691837, -0.0030990, 0.0029847
3: -0.0069931, 0.0178778, -0.0070101, 0.0190106, -0.0228579, 0.0220143
4: -0.0020527, -0.0001612, -0.0021389, -0.0001599, -0.0016743, 0.0017385
5: 0.0151957, 0.0171075, 0.0151086, 0.0171088, -0.0016922, 0.0017570
6: 0.0035857, 0.0045156, 0.0035851, 0.0045580, -0.0008546, 0.0008231
7: -0.0124114, -0.0059659, -0.0127050, -0.0059615, -0.0057052, 0.0059238
8: 0.0068825, 0.0119961, 0.0066496, 0.0119996, -0.0045262, 0.0046997
9: 0.0101035, 0.0193007, 0.0096846, 0.0193070, -0.0081408, 0.0084528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019798, upper bound: 0.0020689
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020012, upper bound: 0.0020012
time: 1.53 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041725, -0.0040883, -0.0041749, -0.0040998, -0.0000632, 0.0000763
1: -0.0089153, -0.0057622, -0.0090040, -0.0061942, -0.0023675, 0.0028572
2: 0.9657646, 0.9695485, 0.9656582, 0.9690302, -0.0028410, 0.0034288
3: -0.0062079, 0.0217011, -0.0069931, 0.0178778, -0.0209551, 0.0252903
4: -0.0023435, -0.0002209, -0.0020527, -0.0001612, -0.0019235, 0.0015938
5: 0.0149018, 0.0170471, 0.0151957, 0.0171075, -0.0019440, 0.0016108
6: 0.0036151, 0.0046586, 0.0035857, 0.0045156, -0.0007835, 0.0009456
7: -0.0134023, -0.0061694, -0.0124114, -0.0059659, -0.0065542, 0.0054307
8: 0.0060964, 0.0118346, 0.0068825, 0.0119961, -0.0051998, 0.0043085
9: 0.0086896, 0.0190103, 0.0101035, 0.0193007, -0.0093523, 0.0077492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018791, upper bound: 0.0018928
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018762, upper bound: 0.0017948
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041725, -0.0040883, -0.0041749, -0.0040964, -0.0000682, 0.0000788
1: -0.0089153, -0.0057622, -0.0090060, -0.0060662, -0.0025555, 0.0029499
2: 0.9657646, 0.9695485, 0.9656559, 0.9691837, -0.0030667, 0.0035400
3: -0.0062079, 0.0217011, -0.0070101, 0.0190106, -0.0226192, 0.0261107
4: -0.0023435, -0.0002209, -0.0021389, -0.0001599, -0.0019859, 0.0017203
5: 0.0149018, 0.0170471, 0.0151086, 0.0171088, -0.0020071, 0.0017387
6: 0.0036151, 0.0046586, 0.0035851, 0.0045580, -0.0008457, 0.0009762
7: -0.0134023, -0.0061694, -0.0127050, -0.0059615, -0.0067668, 0.0058620
8: 0.0060964, 0.0118346, 0.0066496, 0.0119996, -0.0053685, 0.0046506
9: 0.0086896, 0.0190103, 0.0096846, 0.0193070, -0.0096557, 0.0083645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018791, upper bound: 0.0018928
time: 1.57 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018762, upper bound: 0.0017948
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041749, -0.0040964, -0.0041749, -0.0040998, -0.0000664, 0.0000690
1: -0.0090060, -0.0060662, -0.0090040, -0.0061942, -0.0024871, 0.0025824
2: 0.9656559, 0.9691837, 0.9656582, 0.9690302, -0.0029847, 0.0030990
3: -0.0070101, 0.0190106, -0.0069931, 0.0178778, -0.0220143, 0.0228579
4: -0.0021389, -0.0001599, -0.0020527, -0.0001612, -0.0017385, 0.0016743
5: 0.0151086, 0.0171088, 0.0151957, 0.0171075, -0.0017570, 0.0016922
6: 0.0035851, 0.0045580, 0.0035857, 0.0045156, -0.0008231, 0.0008546
7: -0.0127050, -0.0059615, -0.0124114, -0.0059659, -0.0059238, 0.0057052
8: 0.0066496, 0.0119996, 0.0068825, 0.0119961, -0.0046997, 0.0045262
9: 0.0096846, 0.0193070, 0.0101035, 0.0193007, -0.0084528, 0.0081408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018059, upper bound: 0.0020305
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017948, upper bound: 0.0019287
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041749, -0.0040964, -0.0041725, -0.0040883, -0.0000788, 0.0000682
1: -0.0090060, -0.0060662, -0.0089153, -0.0057622, -0.0029499, 0.0025555
2: 0.9656559, 0.9691837, 0.9657646, 0.9695485, -0.0035400, 0.0030667
3: -0.0070101, 0.0190106, -0.0062079, 0.0217011, -0.0261107, 0.0226192
4: -0.0021389, -0.0001599, -0.0023435, -0.0002209, -0.0017203, 0.0019859
5: 0.0151086, 0.0171088, 0.0149018, 0.0170471, -0.0017387, 0.0020071
6: 0.0035851, 0.0045580, 0.0036151, 0.0046586, -0.0009762, 0.0008457
7: -0.0127050, -0.0059615, -0.0134023, -0.0061694, -0.0058620, 0.0067668
8: 0.0066496, 0.0119996, 0.0060964, 0.0118346, -0.0046506, 0.0053685
9: 0.0096846, 0.0193070, 0.0086896, 0.0190103, -0.0083645, 0.0096557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018059, upper bound: 0.0020306
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017948, upper bound: 0.0019287
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041749, -0.0040964, -0.0041749, -0.0040964, -0.0000660, 0.0000660
1: -0.0090060, -0.0060662, -0.0090060, -0.0060662, -0.0024714, 0.0024714
2: 0.9656559, 0.9691837, 0.9656559, 0.9691837, -0.0029658, 0.0029658
3: -0.0070101, 0.0190106, -0.0070101, 0.0190106, -0.0218755, 0.0218755
4: -0.0021389, -0.0001599, -0.0021389, -0.0001599, -0.0016638, 0.0016638
5: 0.0151086, 0.0171088, 0.0151086, 0.0171088, -0.0016815, 0.0016815
6: 0.0035851, 0.0045580, 0.0035851, 0.0045580, -0.0008179, 0.0008179
7: -0.0127050, -0.0059615, -0.0127050, -0.0059615, -0.0056692, 0.0056692
8: 0.0066496, 0.0119996, 0.0066496, 0.0119996, -0.0044977, 0.0044977
9: 0.0096846, 0.0193070, 0.0096846, 0.0193070, -0.0080896, 0.0080896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018214, upper bound: 0.0020321
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018189, upper bound: 0.0019363
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041749, -0.0040964, -0.0041726, -0.0040848, -0.0000784, 0.0000653
1: -0.0090060, -0.0060662, -0.0089186, -0.0056334, -0.0029358, 0.0024443
2: 0.9656559, 0.9691837, 0.9657608, 0.9697031, -0.0035231, 0.0029333
3: -0.0070101, 0.0190106, -0.0062365, 0.0228413, -0.0259861, 0.0216352
4: -0.0021389, -0.0001599, -0.0024302, -0.0002187, -0.0016455, 0.0019764
5: 0.0151086, 0.0171088, 0.0148141, 0.0170493, -0.0016630, 0.0019975
6: 0.0035851, 0.0045580, 0.0036140, 0.0047012, -0.0009716, 0.0008089
7: -0.0127050, -0.0059615, -0.0136978, -0.0061620, -0.0056069, 0.0067345
8: 0.0066496, 0.0119996, 0.0058620, 0.0118405, -0.0044483, 0.0053428
9: 0.0096846, 0.0193070, 0.0082679, 0.0190209, -0.0080007, 0.0096096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018214, upper bound: 0.0020326
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018189, upper bound: 0.0019363
time: 1.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041726, -0.0040848, -0.0041749, -0.0040964, -0.0000653, 0.0000784
1: -0.0089186, -0.0056334, -0.0090060, -0.0060662, -0.0024443, 0.0029358
2: 0.9657608, 0.9697031, 0.9656559, 0.9691837, -0.0029333, 0.0035231
3: -0.0062365, 0.0228413, -0.0070101, 0.0190106, -0.0216352, 0.0259861
4: -0.0024302, -0.0002187, -0.0021389, -0.0001599, -0.0019764, 0.0016455
5: 0.0148141, 0.0170493, 0.0151086, 0.0171088, -0.0019975, 0.0016630
6: 0.0036140, 0.0047012, 0.0035851, 0.0045580, -0.0008089, 0.0009716
7: -0.0136978, -0.0061620, -0.0127050, -0.0059615, -0.0067345, 0.0056069
8: 0.0058620, 0.0118405, 0.0066496, 0.0119996, -0.0053428, 0.0044483
9: 0.0082679, 0.0190209, 0.0096846, 0.0193070, -0.0096096, 0.0080007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018070, upper bound: 0.0019331
time: 1.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017919, upper bound: 0.0018251
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041726, -0.0040848, -0.0041726, -0.0040848, -0.0000716, 0.0000716
1: -0.0089186, -0.0056334, -0.0089186, -0.0056334, -0.0026827, 0.0026827
2: 0.9657608, 0.9697031, 0.9657608, 0.9697031, -0.0032193, 0.0032193
3: -0.0062365, 0.0228413, -0.0062365, 0.0228413, -0.0237453, 0.0237453
4: -0.0024302, -0.0002187, -0.0024302, -0.0002187, -0.0018060, 0.0018060
5: 0.0148141, 0.0170493, 0.0148141, 0.0170493, -0.0018253, 0.0018253
6: 0.0036140, 0.0047012, 0.0036140, 0.0047012, -0.0008878, 0.0008878
7: -0.0136978, -0.0061620, -0.0136978, -0.0061620, -0.0061538, 0.0061538
8: 0.0058620, 0.0118405, 0.0058620, 0.0118405, -0.0048821, 0.0048821
9: 0.0082679, 0.0190209, 0.0082679, 0.0190209, -0.0087810, 0.0087810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018070, upper bound: 0.0019331
time: 1.22 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017919, upper bound: 0.0018251
time: 1.49 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.18 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0019798, upper bound: 0.0020689
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0020012, upper bound: 0.0020013
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0019798, upper bound: 0.0020689
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0020012, upper bound: 0.0020012
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0018791, upper bound: 0.0018928
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0018762, upper bound: 0.0017948
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0018791, upper bound: 0.0018928
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0018762, upper bound: 0.0017948
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0018059, upper bound: 0.0020305
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0017948, upper bound: 0.0019287
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0018059, upper bound: 0.0020306
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0017948, upper bound: 0.0019287
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0018214, upper bound: 0.0020321
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0018189, upper bound: 0.0019363
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0018214, upper bound: 0.0020326
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0018189, upper bound: 0.0019363
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0018070, upper bound: 0.0019331
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0017919, upper bound: 0.0018251
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0018070, upper bound: 0.0019331
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 2, lower bound: -0.0017919, upper bound: 0.0018251

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041726, -0.0041000, -0.0041749, -0.0040998, -0.0000617, 0.0000638
1: -0.0089179, -0.0062002, -0.0090040, -0.0061942, -0.0023114, 0.0023883
2: 0.9657615, 0.9690230, 0.9656582, 0.9690302, -0.0027738, 0.0028660
3: -0.0062308, 0.0178249, -0.0069931, 0.0178778, -0.0204591, 0.0211395
4: -0.0020487, -0.0002191, -0.0020527, -0.0001612, -0.0016078, 0.0015560
5: 0.0151998, 0.0170489, 0.0151957, 0.0171075, -0.0016249, 0.0015726
6: 0.0036142, 0.0045136, 0.0035857, 0.0045156, -0.0007649, 0.0007904
7: -0.0123977, -0.0061635, -0.0124114, -0.0059659, -0.0054785, 0.0053022
8: 0.0068934, 0.0118393, 0.0068825, 0.0119961, -0.0043464, 0.0042065
9: 0.0101230, 0.0190188, 0.0101035, 0.0193007, -0.0078173, 0.0075658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019764, upper bound: 0.0019764
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019764, upper bound: 0.0020013
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041690, -0.0040882, -0.0041737, -0.0040999, -0.0000638, 0.0000752
1: -0.0087835, -0.0057608, -0.0089605, -0.0061981, -0.0023905, 0.0028158
2: 0.9659229, 0.9695503, 0.9657105, 0.9690256, -0.0028687, 0.0033791
3: -0.0050409, 0.0217135, -0.0066081, 0.0178435, -0.0211593, 0.0249233
4: -0.0023445, -0.0003096, -0.0020501, -0.0001904, -0.0018956, 0.0016093
5: 0.0149008, 0.0169574, 0.0151983, 0.0170779, -0.0019158, 0.0016265
6: 0.0036587, 0.0046590, 0.0036001, 0.0045143, -0.0007911, 0.0009318
7: -0.0134055, -0.0064719, -0.0124025, -0.0060657, -0.0064591, 0.0054836
8: 0.0060939, 0.0115947, 0.0068896, 0.0119169, -0.0051243, 0.0043504
9: 0.0086850, 0.0185788, 0.0101161, 0.0191583, -0.0092166, 0.0078247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020012, upper bound: 0.0019764
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020012, upper bound: 0.0020013
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041726, -0.0041000, -0.0041749, -0.0040964, -0.0000667, 0.0000663
1: -0.0089179, -0.0062002, -0.0090060, -0.0060662, -0.0024994, 0.0024810
2: 0.9657615, 0.9690230, 0.9656559, 0.9691837, -0.0029994, 0.0029773
3: -0.0062308, 0.0178249, -0.0070101, 0.0190106, -0.0221232, 0.0219599
4: -0.0020487, -0.0002191, -0.0021389, -0.0001599, -0.0016702, 0.0016826
5: 0.0151998, 0.0170489, 0.0151086, 0.0171088, -0.0016880, 0.0017006
6: 0.0036142, 0.0045136, 0.0035851, 0.0045580, -0.0008272, 0.0008210
7: -0.0123977, -0.0061635, -0.0127050, -0.0059615, -0.0056911, 0.0057334
8: 0.0068934, 0.0118393, 0.0066496, 0.0119996, -0.0045150, 0.0045486
9: 0.0101230, 0.0190188, 0.0096846, 0.0193070, -0.0081207, 0.0081811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020201, upper bound: 0.0019764
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020201, upper bound: 0.0020013
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041690, -0.0040882, -0.0041737, -0.0040965, -0.0000689, 0.0000776
1: -0.0087835, -0.0057608, -0.0089613, -0.0060702, -0.0025786, 0.0029057
2: 0.9659229, 0.9695503, 0.9657095, 0.9691789, -0.0030944, 0.0034869
3: -0.0050409, 0.0217135, -0.0066147, 0.0189750, -0.0228239, 0.0257189
4: -0.0023445, -0.0003096, -0.0021362, -0.0001899, -0.0019561, 0.0017359
5: 0.0149008, 0.0169574, 0.0151113, 0.0170784, -0.0019770, 0.0017544
6: 0.0036587, 0.0046590, 0.0035999, 0.0045566, -0.0008533, 0.0009616
7: -0.0134055, -0.0064719, -0.0126958, -0.0060640, -0.0066653, 0.0059150
8: 0.0060939, 0.0115947, 0.0066569, 0.0119183, -0.0052879, 0.0046927
9: 0.0086850, 0.0185788, 0.0096977, 0.0191607, -0.0095108, 0.0084403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020504, upper bound: 0.0019764
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020504, upper bound: 0.0020013
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041726, -0.0040966, -0.0041749, -0.0040998, -0.0000643, 0.0000688
1: -0.0089213, -0.0060722, -0.0090040, -0.0061942, -0.0024076, 0.0025766
2: 0.9657575, 0.9691767, 0.9656582, 0.9690302, -0.0028892, 0.0030921
3: -0.0062608, 0.0189578, -0.0069931, 0.0178778, -0.0213103, 0.0228064
4: -0.0021349, -0.0002169, -0.0020527, -0.0001612, -0.0017346, 0.0016208
5: 0.0151127, 0.0170512, 0.0151957, 0.0171075, -0.0017531, 0.0016381
6: 0.0036131, 0.0045560, 0.0035857, 0.0045156, -0.0007968, 0.0008527
7: -0.0126913, -0.0061557, -0.0124114, -0.0059659, -0.0059105, 0.0055227
8: 0.0066604, 0.0118455, 0.0068825, 0.0119961, -0.0046891, 0.0043815
9: 0.0097041, 0.0190299, 0.0101035, 0.0193007, -0.0084338, 0.0078805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019764, upper bound: 0.0020201
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019764, upper bound: 0.0020505
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041726, -0.0040966, -0.0041725, -0.0040883, -0.0000767, 0.0000681
1: -0.0089213, -0.0060722, -0.0089153, -0.0057622, -0.0028704, 0.0025496
2: 0.9657575, 0.9691767, 0.9657646, 0.9695485, -0.0034446, 0.0030597
3: -0.0062608, 0.0189578, -0.0062079, 0.0217011, -0.0254067, 0.0225677
4: -0.0021349, -0.0002169, -0.0023435, -0.0002209, -0.0017164, 0.0019323
5: 0.0151127, 0.0170512, 0.0149018, 0.0170471, -0.0017347, 0.0019530
6: 0.0036131, 0.0045560, 0.0036151, 0.0046586, -0.0009499, 0.0008438
7: -0.0126913, -0.0061557, -0.0134023, -0.0061694, -0.0058486, 0.0065844
8: 0.0066604, 0.0118455, 0.0060964, 0.0118346, -0.0046400, 0.0052237
9: 0.0097041, 0.0190299, 0.0086896, 0.0190103, -0.0083455, 0.0093954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017860, upper bound: 0.0019176
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017860, upper bound: 0.0019287
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041726, -0.0040966, -0.0041749, -0.0040964, -0.0000638, 0.0000658
1: -0.0089213, -0.0060722, -0.0090060, -0.0060662, -0.0023885, 0.0024653
2: 0.9657575, 0.9691767, 0.9656559, 0.9691837, -0.0028663, 0.0029585
3: -0.0062608, 0.0189578, -0.0070101, 0.0190106, -0.0211411, 0.0218211
4: -0.0021349, -0.0002169, -0.0021389, -0.0001599, -0.0016596, 0.0016079
5: 0.0151127, 0.0170512, 0.0151086, 0.0171088, -0.0016773, 0.0016251
6: 0.0036131, 0.0045560, 0.0035851, 0.0045580, -0.0007904, 0.0008159
7: -0.0126913, -0.0061557, -0.0127050, -0.0059615, -0.0056551, 0.0054789
8: 0.0066604, 0.0118455, 0.0066496, 0.0119996, -0.0044865, 0.0043467
9: 0.0097041, 0.0190299, 0.0096846, 0.0193070, -0.0080694, 0.0078179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019843, upper bound: 0.0020205
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019843, upper bound: 0.0020529
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041726, -0.0040966, -0.0041726, -0.0040848, -0.0000762, 0.0000651
1: -0.0089213, -0.0060722, -0.0089186, -0.0056334, -0.0028529, 0.0024381
2: 0.9657575, 0.9691767, 0.9657608, 0.9697031, -0.0034236, 0.0029259
3: -0.0062608, 0.0189578, -0.0062365, 0.0228413, -0.0252516, 0.0215807
4: -0.0021349, -0.0002169, -0.0024302, -0.0002187, -0.0016413, 0.0019205
5: 0.0151127, 0.0170512, 0.0148141, 0.0170493, -0.0016589, 0.0019410
6: 0.0036131, 0.0045560, 0.0036140, 0.0047012, -0.0009441, 0.0008069
7: -0.0126913, -0.0061557, -0.0136978, -0.0061620, -0.0055928, 0.0065442
8: 0.0066604, 0.0118455, 0.0058620, 0.0118405, -0.0044371, 0.0051918
9: 0.0097041, 0.0190299, 0.0082679, 0.0190209, -0.0079805, 0.0093380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018066, upper bound: 0.0019222
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018066, upper bound: 0.0019363
time: 1.03 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.55 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.55
Output dim: 2, lower bound: -0.0019764, upper bound: 0.0019764
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 2, lower bound: -0.0019764, upper bound: 0.0020013
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 2, lower bound: -0.0020012, upper bound: 0.0019764
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 2, lower bound: -0.0020012, upper bound: 0.0020013
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 2, lower bound: -0.0020201, upper bound: 0.0019764
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 2, lower bound: -0.0020201, upper bound: 0.0020013
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 2, lower bound: -0.0020504, upper bound: 0.0019764
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 2, lower bound: -0.0020504, upper bound: 0.0020013
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 2, lower bound: -0.0019764, upper bound: 0.0020201
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 2, lower bound: -0.0019764, upper bound: 0.0020505
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.55
Output dim: 2, lower bound: -0.0017860, upper bound: 0.0019176
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.55
Output dim: 2, lower bound: -0.0017860, upper bound: 0.0019287
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 2, lower bound: -0.0019843, upper bound: 0.0020205
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 2, lower bound: -0.0019843, upper bound: 0.0020529
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.55
Output dim: 2, lower bound: -0.0018066, upper bound: 0.0019222
IS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.55
Output dim: 2, lower bound: -0.0018066, upper bound: 0.0019363

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041726, -0.0041000, -0.0041690, -0.0040882, -0.0000741, 0.0000594
1: -0.0089179, -0.0062002, -0.0087835, -0.0057608, -0.0027755, 0.0022253
2: 0.9657615, 0.9690230, 0.9659229, 0.9695503, -0.0033307, 0.0026704
3: -0.0062308, 0.0178249, -0.0050409, 0.0217135, -0.0245667, 0.0196966
4: -0.0020487, -0.0002191, -0.0023445, -0.0003096, -0.0014980, 0.0018684
5: 0.0151998, 0.0170489, 0.0149008, 0.0169574, -0.0015140, 0.0018884
6: 0.0036142, 0.0045136, 0.0036587, 0.0046590, -0.0009185, 0.0007364
7: -0.0123977, -0.0061635, -0.0134055, -0.0064719, -0.0051045, 0.0063667
8: 0.0068934, 0.0118393, 0.0060939, 0.0115947, -0.0040497, 0.0050510
9: 0.0101230, 0.0190188, 0.0086850, 0.0185788, -0.0072838, 0.0090848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018621, upper bound: 0.0019899
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018708, upper bound: 0.0019791
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041690, -0.0040882, -0.0041726, -0.0041000, -0.0000594, 0.0000741
1: -0.0087835, -0.0057608, -0.0089179, -0.0062002, -0.0022253, 0.0027755
2: 0.9659229, 0.9695503, 0.9657615, 0.9690230, -0.0026704, 0.0033307
3: -0.0050409, 0.0217135, -0.0062308, 0.0178249, -0.0196966, 0.0245667
4: -0.0023445, -0.0003096, -0.0020487, -0.0002191, -0.0018684, 0.0014980
5: 0.0149008, 0.0169574, 0.0151998, 0.0170489, -0.0018884, 0.0015140
6: 0.0036587, 0.0046590, 0.0036142, 0.0045136, -0.0007364, 0.0009185
7: -0.0134055, -0.0064719, -0.0123977, -0.0061635, -0.0063667, 0.0051045
8: 0.0060939, 0.0115947, 0.0068934, 0.0118393, -0.0050510, 0.0040497
9: 0.0086850, 0.0185788, 0.0101230, 0.0190188, -0.0090848, 0.0072838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018672, upper bound: 0.0018910
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018688, upper bound: 0.0018571
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041690, -0.0040882, -0.0041690, -0.0040882, -0.0000650, 0.0000650
1: -0.0087835, -0.0057608, -0.0087835, -0.0057608, -0.0024347, 0.0024347
2: 0.9659229, 0.9695503, 0.9659229, 0.9695503, -0.0029217, 0.0029217
3: -0.0050409, 0.0217135, -0.0050409, 0.0217135, -0.0215500, 0.0215500
4: -0.0023445, -0.0003096, -0.0023445, -0.0003096, -0.0016390, 0.0016390
5: 0.0149008, 0.0169574, 0.0149008, 0.0169574, -0.0016565, 0.0016565
6: 0.0036587, 0.0046590, 0.0036587, 0.0046590, -0.0008057, 0.0008057
7: -0.0134055, -0.0064719, -0.0134055, -0.0064719, -0.0055849, 0.0055849
8: 0.0060939, 0.0115947, 0.0060939, 0.0115947, -0.0044308, 0.0044308
9: 0.0086850, 0.0185788, 0.0086850, 0.0185788, -0.0079692, 0.0079692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018672, upper bound: 0.0018910
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018688, upper bound: 0.0018571
time: 1.47 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041726, -0.0041000, -0.0041726, -0.0040966, -0.0000666, 0.0000641
1: -0.0089179, -0.0062002, -0.0089213, -0.0060722, -0.0024936, 0.0024014
2: 0.9657615, 0.9690230, 0.9657575, 0.9691767, -0.0029924, 0.0028818
3: -0.0062308, 0.0178249, -0.0062608, 0.0189578, -0.0220717, 0.0212559
4: -0.0020487, -0.0002191, -0.0021349, -0.0002169, -0.0016166, 0.0016787
5: 0.0151998, 0.0170489, 0.0151127, 0.0170512, -0.0016339, 0.0016966
6: 0.0036142, 0.0045136, 0.0036131, 0.0045560, -0.0008252, 0.0007947
7: -0.0123977, -0.0061635, -0.0126913, -0.0061557, -0.0055086, 0.0057201
8: 0.0068934, 0.0118393, 0.0066604, 0.0118455, -0.0043703, 0.0045380
9: 0.0101230, 0.0190188, 0.0097041, 0.0190299, -0.0078604, 0.0081621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019009, upper bound: 0.0019838
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019143, upper bound: 0.0019788
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041726, -0.0041000, -0.0041689, -0.0040852, -0.0000788, 0.0000611
1: -0.0089179, -0.0062002, -0.0087826, -0.0056457, -0.0029491, 0.0022897
2: 0.9657615, 0.9690230, 0.9659240, 0.9696884, -0.0035391, 0.0027477
3: -0.0062308, 0.0178249, -0.0050328, 0.0227327, -0.0261037, 0.0202667
4: -0.0020487, -0.0002191, -0.0024220, -0.0003103, -0.0015414, 0.0019853
5: 0.0151998, 0.0170489, 0.0148225, 0.0169568, -0.0015579, 0.0020065
6: 0.0036142, 0.0045136, 0.0036590, 0.0046971, -0.0009760, 0.0007577
7: -0.0123977, -0.0061635, -0.0136696, -0.0064739, -0.0052523, 0.0067650
8: 0.0068934, 0.0118393, 0.0058843, 0.0115930, -0.0041669, 0.0053670
9: 0.0101230, 0.0190188, 0.0083081, 0.0185758, -0.0074946, 0.0096531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019009, upper bound: 0.0019898
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019143, upper bound: 0.0019791
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041690, -0.0040882, -0.0041726, -0.0040966, -0.0000645, 0.0000767
1: -0.0087835, -0.0057608, -0.0089213, -0.0060722, -0.0024136, 0.0028717
2: 0.9659229, 0.9695503, 0.9657575, 0.9691767, -0.0028964, 0.0034461
3: -0.0050409, 0.0217135, -0.0062608, 0.0189578, -0.0213635, 0.0254179
4: -0.0023445, -0.0003096, -0.0021349, -0.0002169, -0.0019332, 0.0016248
5: 0.0149008, 0.0169574, 0.0151127, 0.0170512, -0.0019538, 0.0016422
6: 0.0036587, 0.0046590, 0.0036131, 0.0045560, -0.0007987, 0.0009503
7: -0.0134055, -0.0064719, -0.0126913, -0.0061557, -0.0065873, 0.0055365
8: 0.0060939, 0.0115947, 0.0066604, 0.0118455, -0.0052260, 0.0043924
9: 0.0086850, 0.0185788, 0.0097041, 0.0190299, -0.0093995, 0.0079002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019076, upper bound: 0.0018910
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019122, upper bound: 0.0018571
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041690, -0.0040882, -0.0041689, -0.0040852, -0.0000702, 0.0000676
1: -0.0087835, -0.0057608, -0.0087826, -0.0056457, -0.0026271, 0.0025312
2: 0.9659229, 0.9695503, 0.9659240, 0.9696884, -0.0031526, 0.0030375
3: -0.0050409, 0.0217135, -0.0050328, 0.0227327, -0.0232534, 0.0224042
4: -0.0023445, -0.0003096, -0.0024220, -0.0003103, -0.0017040, 0.0017686
5: 0.0149008, 0.0169574, 0.0148225, 0.0169568, -0.0017222, 0.0017874
6: 0.0036587, 0.0046590, 0.0036590, 0.0046971, -0.0008694, 0.0008377
7: -0.0134055, -0.0064719, -0.0136696, -0.0064739, -0.0058062, 0.0060263
8: 0.0060939, 0.0115947, 0.0058843, 0.0115930, -0.0046064, 0.0047810
9: 0.0086850, 0.0185788, 0.0083081, 0.0185758, -0.0082850, 0.0085991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019077, upper bound: 0.0018910
time: 1.30 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019122, upper bound: 0.0018571
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041726, -0.0040966, -0.0041726, -0.0041000, -0.0000641, 0.0000666
1: -0.0089213, -0.0060722, -0.0089179, -0.0062002, -0.0024014, 0.0024936
2: 0.9657575, 0.9691767, 0.9657615, 0.9690230, -0.0028818, 0.0029924
3: -0.0062608, 0.0189578, -0.0062308, 0.0178249, -0.0212559, 0.0220717
4: -0.0021349, -0.0002169, -0.0020487, -0.0002191, -0.0016787, 0.0016166
5: 0.0151127, 0.0170512, 0.0151998, 0.0170489, -0.0016966, 0.0016339
6: 0.0036131, 0.0045560, 0.0036142, 0.0045136, -0.0007947, 0.0008252
7: -0.0126913, -0.0061557, -0.0123977, -0.0061635, -0.0057201, 0.0055086
8: 0.0066604, 0.0118455, 0.0068934, 0.0118393, -0.0045380, 0.0043703
9: 0.0097041, 0.0190299, 0.0101230, 0.0190188, -0.0081621, 0.0078604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018621, upper bound: 0.0020164
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018708, upper bound: 0.0020115
time: 1.31 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041726, -0.0040966, -0.0041690, -0.0040882, -0.0000767, 0.0000645
1: -0.0089213, -0.0060722, -0.0087835, -0.0057608, -0.0028717, 0.0024136
2: 0.9657575, 0.9691767, 0.9659229, 0.9695503, -0.0034461, 0.0028964
3: -0.0062608, 0.0189578, -0.0050409, 0.0217135, -0.0254179, 0.0213635
4: -0.0021349, -0.0002169, -0.0023445, -0.0003096, -0.0016248, 0.0019332
5: 0.0151127, 0.0170512, 0.0149008, 0.0169574, -0.0016422, 0.0019538
6: 0.0036131, 0.0045560, 0.0036587, 0.0046590, -0.0009503, 0.0007987
7: -0.0126913, -0.0061557, -0.0134055, -0.0064719, -0.0055365, 0.0065873
8: 0.0066604, 0.0118455, 0.0060939, 0.0115947, -0.0043924, 0.0052260
9: 0.0097041, 0.0190299, 0.0086850, 0.0185788, -0.0079002, 0.0093995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018621, upper bound: 0.0020229
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018708, upper bound: 0.0020120
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041726, -0.0040966, -0.0041726, -0.0040966, -0.0000636, 0.0000636
1: -0.0089213, -0.0060722, -0.0089213, -0.0060722, -0.0023823, 0.0023823
2: 0.9657575, 0.9691767, 0.9657575, 0.9691767, -0.0028589, 0.0028589
3: -0.0062608, 0.0189578, -0.0062608, 0.0189578, -0.0210866, 0.0210866
4: -0.0021349, -0.0002169, -0.0021349, -0.0002169, -0.0016038, 0.0016038
5: 0.0151127, 0.0170512, 0.0151127, 0.0170512, -0.0016209, 0.0016209
6: 0.0036131, 0.0045560, 0.0036131, 0.0045560, -0.0007884, 0.0007884
7: -0.0126913, -0.0061557, -0.0126913, -0.0061557, -0.0054648, 0.0054648
8: 0.0066604, 0.0118455, 0.0066604, 0.0118455, -0.0043355, 0.0043355
9: 0.0097041, 0.0190299, 0.0097041, 0.0190299, -0.0077978, 0.0077978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018687, upper bound: 0.0020164
time: 1.32 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018781, upper bound: 0.0020116
time: 1.25 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041726, -0.0040966, -0.0041689, -0.0040852, -0.0000761, 0.0000615
1: -0.0089213, -0.0060722, -0.0087826, -0.0056457, -0.0028485, 0.0023024
2: 0.9657575, 0.9691767, 0.9659240, 0.9696884, -0.0034184, 0.0027630
3: -0.0062608, 0.0189578, -0.0050328, 0.0227327, -0.0252132, 0.0203792
4: -0.0021349, -0.0002169, -0.0024220, -0.0003103, -0.0015500, 0.0019176
5: 0.0151127, 0.0170512, 0.0148225, 0.0169568, -0.0015665, 0.0019381
6: 0.0036131, 0.0045560, 0.0036590, 0.0046971, -0.0009427, 0.0007619
7: -0.0126913, -0.0061557, -0.0136696, -0.0064739, -0.0052815, 0.0065342
8: 0.0066604, 0.0118455, 0.0058843, 0.0115930, -0.0041901, 0.0051839
9: 0.0097041, 0.0190299, 0.0083081, 0.0185758, -0.0075362, 0.0093238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018687, upper bound: 0.0020232
time: 1.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018781, upper bound: 0.0020121
time: 1.17 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.35 seconds
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0018621, upper bound: 0.0019899
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0018708, upper bound: 0.0019791
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0018672, upper bound: 0.0018910
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0018688, upper bound: 0.0018571
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0018672, upper bound: 0.0018910
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0018688, upper bound: 0.0018571
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0019009, upper bound: 0.0019838
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0019143, upper bound: 0.0019788
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0019009, upper bound: 0.0019898
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0019143, upper bound: 0.0019791
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0019076, upper bound: 0.0018910
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0019122, upper bound: 0.0018571
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0019077, upper bound: 0.0018910
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0019122, upper bound: 0.0018571
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0018621, upper bound: 0.0020164
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0018708, upper bound: 0.0020115
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0018621, upper bound: 0.0020229
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0018708, upper bound: 0.0020120
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0018687, upper bound: 0.0020164
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0018781, upper bound: 0.0020116
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0018687, upper bound: 0.0020232
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.35
Output dim: 2, lower bound: -0.0018781, upper bound: 0.0020121

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041707, -0.0040972, -0.0041726, -0.0041000, -0.0000621, 0.0000659
1: -0.0088481, -0.0060969, -0.0089179, -0.0062002, -0.0023260, 0.0024676
2: 0.9658453, 0.9691469, 0.9657615, 0.9690230, -0.0027913, 0.0029612
3: -0.0056128, 0.0187389, -0.0062308, 0.0178249, -0.0205879, 0.0218416
4: -0.0021182, -0.0002661, -0.0020487, -0.0002191, -0.0016612, 0.0015658
5: 0.0151295, 0.0170014, 0.0151998, 0.0170489, -0.0016789, 0.0015826
6: 0.0036373, 0.0045478, 0.0036142, 0.0045136, -0.0007697, 0.0008166
7: -0.0126346, -0.0063237, -0.0123977, -0.0061635, -0.0056605, 0.0053355
8: 0.0067055, 0.0117123, 0.0068934, 0.0118393, -0.0044907, 0.0042330
9: 0.0097850, 0.0187902, 0.0101230, 0.0190188, -0.0080770, 0.0076134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020300, upper bound: 0.0020613
time: 1.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020300, upper bound: 0.0021020
time: 1.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041711, -0.0040956, -0.0041721, -0.0041002, -0.0000625, 0.0000671
1: -0.0088624, -0.0060377, -0.0089027, -0.0062073, -0.0023394, 0.0025135
2: 0.9658282, 0.9692180, 0.9657798, 0.9690144, -0.0028074, 0.0030164
3: -0.0057391, 0.0192624, -0.0060958, 0.0177618, -0.0207070, 0.0222482
4: -0.0021580, -0.0002565, -0.0020439, -0.0002294, -0.0016921, 0.0015749
5: 0.0150893, 0.0170111, 0.0152046, 0.0170385, -0.0017102, 0.0015917
6: 0.0036326, 0.0045674, 0.0036193, 0.0045113, -0.0007742, 0.0008318
7: -0.0127703, -0.0062909, -0.0123814, -0.0061985, -0.0057658, 0.0053664
8: 0.0065978, 0.0117382, 0.0069064, 0.0118116, -0.0045743, 0.0042575
9: 0.0095914, 0.0188370, 0.0101464, 0.0189689, -0.0082273, 0.0076574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020211, upper bound: 0.0019792
time: 1.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020353, upper bound: 0.0020703
time: 1.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041707, -0.0040972, -0.0041690, -0.0040882, -0.0000745, 0.0000638
1: -0.0088481, -0.0060969, -0.0087835, -0.0057608, -0.0027887, 0.0023876
2: 0.9658453, 0.9691469, 0.9659229, 0.9695503, -0.0033465, 0.0028652
3: -0.0056128, 0.0187389, -0.0050409, 0.0217135, -0.0246834, 0.0211335
4: -0.0021182, -0.0002661, -0.0023445, -0.0003096, -0.0016073, 0.0018773
5: 0.0151295, 0.0170014, 0.0149008, 0.0169574, -0.0016245, 0.0018974
6: 0.0036373, 0.0045478, 0.0036587, 0.0046590, -0.0009229, 0.0007901
7: -0.0126346, -0.0063237, -0.0134055, -0.0064719, -0.0054769, 0.0063969
8: 0.0067055, 0.0117123, 0.0060939, 0.0115947, -0.0043451, 0.0050750
9: 0.0097850, 0.0187902, 0.0086850, 0.0185788, -0.0078151, 0.0091279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018621, upper bound: 0.0019923
time: 1.33 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018621, upper bound: 0.0020120
time: 1.30 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041711, -0.0040956, -0.0041686, -0.0040884, -0.0000748, 0.0000649
1: -0.0088624, -0.0060377, -0.0087684, -0.0057675, -0.0028029, 0.0024314
2: 0.9658282, 0.9692180, 0.9659410, 0.9695423, -0.0033636, 0.0029177
3: -0.0057391, 0.0192624, -0.0049076, 0.0216549, -0.0248091, 0.0215207
4: -0.0021580, -0.0002565, -0.0023400, -0.0003198, -0.0016368, 0.0018869
5: 0.0150893, 0.0170111, 0.0149054, 0.0169471, -0.0016543, 0.0019070
6: 0.0036326, 0.0045674, 0.0036637, 0.0046568, -0.0009276, 0.0008046
7: -0.0127703, -0.0062909, -0.0133903, -0.0065064, -0.0055773, 0.0064295
8: 0.0065978, 0.0117382, 0.0061059, 0.0115673, -0.0044248, 0.0051009
9: 0.0095914, 0.0188370, 0.0087067, 0.0185295, -0.0079583, 0.0091744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017769, upper bound: 0.0018434
time: 1.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018409, upper bound: 0.0019797
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041707, -0.0040972, -0.0041726, -0.0040966, -0.0000616, 0.0000629
1: -0.0088481, -0.0060969, -0.0089213, -0.0060722, -0.0023060, 0.0023566
2: 0.9658453, 0.9691469, 0.9657575, 0.9691767, -0.0027673, 0.0028280
3: -0.0056128, 0.0187389, -0.0062608, 0.0189578, -0.0204109, 0.0208589
4: -0.0021182, -0.0002661, -0.0021349, -0.0002169, -0.0015864, 0.0015524
5: 0.0151295, 0.0170014, 0.0151127, 0.0170512, -0.0016034, 0.0015689
6: 0.0036373, 0.0045478, 0.0036131, 0.0045560, -0.0007631, 0.0007799
7: -0.0126346, -0.0063237, -0.0126913, -0.0061557, -0.0054058, 0.0052897
8: 0.0067055, 0.0117123, 0.0066604, 0.0118455, -0.0042887, 0.0041966
9: 0.0097850, 0.0187902, 0.0097041, 0.0190299, -0.0077136, 0.0075479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020296, upper bound: 0.0020613
time: 1.42 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020296, upper bound: 0.0021020
time: 1.38 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041711, -0.0040956, -0.0041723, -0.0040968, -0.0000620, 0.0000640
1: -0.0088624, -0.0060377, -0.0089068, -0.0060795, -0.0023205, 0.0023981
2: 0.9658282, 0.9692180, 0.9657749, 0.9691679, -0.0027848, 0.0028778
3: -0.0057391, 0.0192624, -0.0061325, 0.0188932, -0.0205399, 0.0212264
4: -0.0021580, -0.0002565, -0.0021300, -0.0002266, -0.0016144, 0.0015622
5: 0.0150893, 0.0170111, 0.0151176, 0.0170413, -0.0016316, 0.0015789
6: 0.0036326, 0.0045674, 0.0036179, 0.0045536, -0.0007680, 0.0007936
7: -0.0127703, -0.0062909, -0.0126746, -0.0061890, -0.0055010, 0.0053231
8: 0.0065978, 0.0117382, 0.0066737, 0.0118191, -0.0043642, 0.0042231
9: 0.0095914, 0.0188370, 0.0097280, 0.0189824, -0.0078495, 0.0075956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020211, upper bound: 0.0019792
time: 1.29 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020351, upper bound: 0.0020703
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041707, -0.0040972, -0.0041689, -0.0040852, -0.0000738, 0.0000608
1: -0.0088481, -0.0060969, -0.0087826, -0.0056457, -0.0027650, 0.0022767
2: 0.9658453, 0.9691469, 0.9659240, 0.9696884, -0.0033181, 0.0027321
3: -0.0056128, 0.0187389, -0.0050328, 0.0227327, -0.0244740, 0.0201515
4: -0.0021182, -0.0002661, -0.0024220, -0.0003103, -0.0015326, 0.0018614
5: 0.0151295, 0.0170014, 0.0148225, 0.0169568, -0.0015490, 0.0018813
6: 0.0036373, 0.0045478, 0.0036590, 0.0046971, -0.0009150, 0.0007534
7: -0.0126346, -0.0063237, -0.0136696, -0.0064739, -0.0052224, 0.0063426
8: 0.0067055, 0.0117123, 0.0058843, 0.0115930, -0.0041432, 0.0050320
9: 0.0097850, 0.0187902, 0.0083081, 0.0185758, -0.0074520, 0.0090505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018687, upper bound: 0.0019924
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018687, upper bound: 0.0020121
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041711, -0.0040956, -0.0041685, -0.0040854, -0.0000742, 0.0000618
1: -0.0088624, -0.0060377, -0.0087674, -0.0056525, -0.0027803, 0.0023161
2: 0.9658282, 0.9692180, 0.9659421, 0.9696802, -0.0033365, 0.0027794
3: -0.0057391, 0.0192624, -0.0048989, 0.0226727, -0.0246096, 0.0205002
4: -0.0021580, -0.0002565, -0.0024174, -0.0003204, -0.0015592, 0.0018717
5: 0.0150893, 0.0170111, 0.0148271, 0.0169465, -0.0015758, 0.0018917
6: 0.0036326, 0.0045674, 0.0036640, 0.0046949, -0.0009201, 0.0007665
7: -0.0127703, -0.0062909, -0.0136541, -0.0065087, -0.0053128, 0.0063778
8: 0.0065978, 0.0117382, 0.0058966, 0.0115655, -0.0042149, 0.0050598
9: 0.0095914, 0.0188370, 0.0083303, 0.0185262, -0.0075809, 0.0091006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017983, upper bound: 0.0018544
time: 1.38 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018470, upper bound: 0.0019796
time: 1.25 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.12 seconds
IS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0020300, upper bound: 0.0020613
IS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0020300, upper bound: 0.0021020
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0020211, upper bound: 0.0019792
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0020353, upper bound: 0.0020703
IS_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0018621, upper bound: 0.0019923
IS_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0018621, upper bound: 0.0020120
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0017769, upper bound: 0.0018434
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0018409, upper bound: 0.0019797
IS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0020296, upper bound: 0.0020613
IS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0020296, upper bound: 0.0021020
IS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0020211, upper bound: 0.0019792
IS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0020351, upper bound: 0.0020703
IS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0018687, upper bound: 0.0019924
IS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0018687, upper bound: 0.0020121
IS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0017983, upper bound: 0.0018544
IS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0018470, upper bound: 0.0019796

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041707, -0.0040972, -0.0041706, -0.0041006, -0.0000614, 0.0000639
1: -0.0088481, -0.0060969, -0.0088450, -0.0062246, -0.0022998, 0.0023916
2: 0.9658453, 0.9691469, 0.9658492, 0.9689937, -0.0027598, 0.0028701
3: -0.0056128, 0.0187389, -0.0055850, 0.0176087, -0.0203558, 0.0211690
4: -0.0021182, -0.0002661, -0.0020323, -0.0002683, -0.0016100, 0.0015482
5: 0.0151295, 0.0170014, 0.0152164, 0.0169992, -0.0016272, 0.0015647
6: 0.0036373, 0.0045478, 0.0036384, 0.0045055, -0.0007611, 0.0007915
7: -0.0126346, -0.0063237, -0.0123417, -0.0063308, -0.0054861, 0.0052754
8: 0.0067055, 0.0117123, 0.0069378, 0.0117065, -0.0043524, 0.0041852
9: 0.0097850, 0.0187902, 0.0102030, 0.0187800, -0.0078283, 0.0075276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019326, upper bound: 0.0020415
time: 1.23 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019947, upper bound: 0.0020498
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041707, -0.0040972, -0.0041708, -0.0040992, -0.0000629, 0.0000644
1: -0.0088481, -0.0060969, -0.0088538, -0.0061703, -0.0023568, 0.0024123
2: 0.9658453, 0.9691469, 0.9658386, 0.9690588, -0.0028282, 0.0028949
3: -0.0056128, 0.0187389, -0.0056637, 0.0180892, -0.0208606, 0.0213523
4: -0.0021182, -0.0002661, -0.0020688, -0.0002623, -0.0016240, 0.0015866
5: 0.0151295, 0.0170014, 0.0151794, 0.0170053, -0.0016413, 0.0016035
6: 0.0036373, 0.0045478, 0.0036354, 0.0045235, -0.0007799, 0.0007983
7: -0.0126346, -0.0063237, -0.0124662, -0.0063104, -0.0055336, 0.0054062
8: 0.0067055, 0.0117123, 0.0068390, 0.0117227, -0.0043901, 0.0042890
9: 0.0097850, 0.0187902, 0.0100253, 0.0188091, -0.0078961, 0.0077142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019326, upper bound: 0.0020568
time: 1.23 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019947, upper bound: 0.0020703
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041705, -0.0040957, -0.0041695, -0.0040998, -0.0000626, 0.0000644
1: -0.0088417, -0.0060402, -0.0088020, -0.0061943, -0.0023454, 0.0024135
2: 0.9658530, 0.9692150, 0.9659007, 0.9690301, -0.0028146, 0.0028963
3: -0.0055564, 0.0192405, -0.0052048, 0.0178766, -0.0207598, 0.0213624
4: -0.0021564, -0.0002704, -0.0020526, -0.0002972, -0.0016247, 0.0015789
5: 0.0150909, 0.0169970, 0.0151958, 0.0169700, -0.0016421, 0.0015958
6: 0.0036394, 0.0045666, 0.0036526, 0.0045156, -0.0007762, 0.0007987
7: -0.0127646, -0.0063383, -0.0124111, -0.0064294, -0.0055362, 0.0053801
8: 0.0066023, 0.0117007, 0.0068827, 0.0116284, -0.0043922, 0.0042683
9: 0.0095995, 0.0187694, 0.0101039, 0.0186394, -0.0078998, 0.0076770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019591, upper bound: 0.0019231
time: 1.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019806, upper bound: 0.0019395
time: 1.33 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041711, -0.0040956, -0.0041714, -0.0041003, -0.0000624, 0.0000656
1: -0.0088624, -0.0060377, -0.0088762, -0.0062109, -0.0023361, 0.0024554
2: 0.9658282, 0.9692180, 0.9658116, 0.9690102, -0.0028034, 0.0029466
3: -0.0057391, 0.0192624, -0.0058616, 0.0177299, -0.0206774, 0.0217333
4: -0.0021580, -0.0002565, -0.0020415, -0.0002472, -0.0016529, 0.0015726
5: 0.0150893, 0.0170111, 0.0152071, 0.0170205, -0.0016706, 0.0015894
6: 0.0036326, 0.0045674, 0.0036280, 0.0045101, -0.0007731, 0.0008126
7: -0.0127703, -0.0062909, -0.0123731, -0.0062592, -0.0056324, 0.0053587
8: 0.0065978, 0.0117382, 0.0069129, 0.0117634, -0.0044684, 0.0042514
9: 0.0095914, 0.0188370, 0.0101581, 0.0188822, -0.0080369, 0.0076465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019513, upper bound: 0.0020568
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019513, upper bound: 0.0020703
time: 1.71 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041707, -0.0040972, -0.0041673, -0.0040874, -0.0000752, 0.0000619
1: -0.0088481, -0.0060969, -0.0087197, -0.0057295, -0.0028171, 0.0023194
2: 0.9658453, 0.9691469, 0.9659994, 0.9695879, -0.0033807, 0.0027833
3: -0.0056128, 0.0187389, -0.0044763, 0.0219908, -0.0249352, 0.0205294
4: -0.0021182, -0.0002661, -0.0023656, -0.0003526, -0.0015614, 0.0018965
5: 0.0151295, 0.0170014, 0.0148795, 0.0169140, -0.0015781, 0.0019167
6: 0.0036373, 0.0045478, 0.0036798, 0.0046694, -0.0009323, 0.0007676
7: -0.0126346, -0.0063237, -0.0134774, -0.0066182, -0.0053204, 0.0064622
8: 0.0067055, 0.0117123, 0.0060369, 0.0114786, -0.0042209, 0.0051268
9: 0.0097850, 0.0187902, 0.0085825, 0.0183700, -0.0075918, 0.0092210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016763, upper bound: 0.0019630
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018318, upper bound: 0.0019898
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041707, -0.0040972, -0.0041707, -0.0040972, -0.0000609, 0.0000609
1: -0.0088481, -0.0060969, -0.0088481, -0.0060969, -0.0022801, 0.0022801
2: 0.9658453, 0.9691469, 0.9658453, 0.9691469, -0.0027362, 0.0027362
3: -0.0056128, 0.0187389, -0.0056128, 0.0187389, -0.0201819, 0.0201819
4: -0.0021182, -0.0002661, -0.0021182, -0.0002661, -0.0015349, 0.0015349
5: 0.0151295, 0.0170014, 0.0151295, 0.0170014, -0.0015513, 0.0015513
6: 0.0036373, 0.0045478, 0.0036373, 0.0045478, -0.0007546, 0.0007546
7: -0.0126346, -0.0063237, -0.0126346, -0.0063237, -0.0052303, 0.0052303
8: 0.0067055, 0.0117123, 0.0067055, 0.0117123, -0.0041495, 0.0041495
9: 0.0097850, 0.0187902, 0.0097850, 0.0187902, -0.0074632, 0.0074632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019299, upper bound: 0.0020415
time: 1.34 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019943, upper bound: 0.0020498
time: 1.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041707, -0.0040972, -0.0041711, -0.0040956, -0.0000624, 0.0000614
1: -0.0088481, -0.0060969, -0.0088624, -0.0060377, -0.0023364, 0.0023002
2: 0.9658453, 0.9691469, 0.9658282, 0.9692180, -0.0028038, 0.0027604
3: -0.0056128, 0.0187389, -0.0057391, 0.0192624, -0.0206802, 0.0203600
4: -0.0021182, -0.0002661, -0.0021580, -0.0002565, -0.0015485, 0.0015729
5: 0.0151295, 0.0170014, 0.0150893, 0.0170111, -0.0015650, 0.0015896
6: 0.0036373, 0.0045478, 0.0036326, 0.0045674, -0.0007732, 0.0007612
7: -0.0126346, -0.0063237, -0.0127703, -0.0062909, -0.0052765, 0.0053595
8: 0.0067055, 0.0117123, 0.0065978, 0.0117382, -0.0041861, 0.0042519
9: 0.0097850, 0.0187902, 0.0095914, 0.0188370, -0.0075291, 0.0076475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019299, upper bound: 0.0020568
time: 1.39 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019943, upper bound: 0.0020703
time: 1.43 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041705, -0.0040957, -0.0041695, -0.0040965, -0.0000621, 0.0000614
1: -0.0088417, -0.0060402, -0.0088036, -0.0060698, -0.0023264, 0.0022977
2: 0.9658530, 0.9692150, 0.9658988, 0.9691794, -0.0027917, 0.0027573
3: -0.0055564, 0.0192405, -0.0052189, 0.0189788, -0.0205913, 0.0203374
4: -0.0021564, -0.0002704, -0.0021365, -0.0002961, -0.0015468, 0.0015661
5: 0.0150909, 0.0169970, 0.0151111, 0.0169711, -0.0015633, 0.0015828
6: 0.0036394, 0.0045666, 0.0036521, 0.0045568, -0.0007699, 0.0007604
7: -0.0127646, -0.0063383, -0.0126968, -0.0064257, -0.0052706, 0.0053364
8: 0.0066023, 0.0117007, 0.0066561, 0.0116313, -0.0041815, 0.0042337
9: 0.0095995, 0.0187694, 0.0096963, 0.0186446, -0.0075208, 0.0076146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019598, upper bound: 0.0019231
time: 1.50 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019807, upper bound: 0.0019395
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041711, -0.0040956, -0.0041716, -0.0040969, -0.0000619, 0.0000626
1: -0.0088624, -0.0060377, -0.0088821, -0.0060833, -0.0023171, 0.0023454
2: 0.9658282, 0.9692180, 0.9658045, 0.9691633, -0.0027806, 0.0028146
3: -0.0057391, 0.0192624, -0.0059137, 0.0188597, -0.0205092, 0.0207601
4: -0.0021580, -0.0002565, -0.0021274, -0.0002433, -0.0015789, 0.0015598
5: 0.0150893, 0.0170111, 0.0151202, 0.0170245, -0.0015958, 0.0015765
6: 0.0036326, 0.0045674, 0.0036261, 0.0045523, -0.0007668, 0.0007762
7: -0.0127703, -0.0062909, -0.0126659, -0.0062457, -0.0053802, 0.0053151
8: 0.0065978, 0.0117382, 0.0066806, 0.0117741, -0.0042684, 0.0042168
9: 0.0095914, 0.0188370, 0.0097403, 0.0189015, -0.0076771, 0.0075843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019504, upper bound: 0.0020568
time: 1.40 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019504, upper bound: 0.0020703
time: 1.33 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041707, -0.0040972, -0.0041672, -0.0040843, -0.0000746, 0.0000590
1: -0.0088481, -0.0060969, -0.0087171, -0.0056122, -0.0027942, 0.0022083
2: 0.9658453, 0.9691469, 0.9660026, 0.9697286, -0.0033532, 0.0026500
3: -0.0056128, 0.0187389, -0.0044530, 0.0230293, -0.0247325, 0.0195461
4: -0.0021182, -0.0002661, -0.0024445, -0.0003544, -0.0014866, 0.0018810
5: 0.0151295, 0.0170014, 0.0147997, 0.0169122, -0.0015025, 0.0019011
6: 0.0036373, 0.0045478, 0.0036807, 0.0047082, -0.0009247, 0.0007308
7: -0.0126346, -0.0063237, -0.0137465, -0.0066242, -0.0050655, 0.0064096
8: 0.0067055, 0.0117123, 0.0058233, 0.0114738, -0.0040188, 0.0050851
9: 0.0097850, 0.0187902, 0.0081984, 0.0183614, -0.0072281, 0.0091460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017040, upper bound: 0.0019645
time: 1.41 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018366, upper bound: 0.0019900
time: 1.09 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 4.03 seconds
IS_A2_B1_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0019326, upper bound: 0.0020415
IS_A2_B1_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0019947, upper bound: 0.0020498
IS_A2_B1_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0019326, upper bound: 0.0020568
IS_A2_B1_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0019947, upper bound: 0.0020703
IS_A2_B1_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0019591, upper bound: 0.0019231
IS_A2_B1_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0019806, upper bound: 0.0019395
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0019513, upper bound: 0.0020568
IS_A2_B1_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0019513, upper bound: 0.0020703
IS_A2_B1_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0016763, upper bound: 0.0019630
IS_A2_B1_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0018318, upper bound: 0.0019898
IS_A2_B2_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0019299, upper bound: 0.0020415
IS_A2_B2_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0019943, upper bound: 0.0020498
IS_A2_B2_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0019299, upper bound: 0.0020568
IS_A2_B2_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0019943, upper bound: 0.0020703
IS_A2_B2_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0019598, upper bound: 0.0019231
IS_A2_B2_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0019807, upper bound: 0.0019395
IS_A2_B2_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0019504, upper bound: 0.0020568
IS_A2_B2_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0019504, upper bound: 0.0020703
IS_A2_B2_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0017040, upper bound: 0.0019645
IS_A2_B2_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.03
Output dim: 2, lower bound: -0.0018366, upper bound: 0.0019900

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041680, -0.0040970, -0.0041701, -0.0041007, -0.0000587, 0.0000639
1: -0.0087459, -0.0060872, -0.0088251, -0.0062271, -0.0021991, 0.0023914
2: 0.9659681, 0.9691586, 0.9658730, 0.9689907, -0.0026391, 0.0028698
3: -0.0047079, 0.0188249, -0.0054091, 0.0175866, -0.0194652, 0.0211670
4: -0.0021248, -0.0003350, -0.0020306, -0.0002816, -0.0016099, 0.0014804
5: 0.0151229, 0.0169318, 0.0152181, 0.0169857, -0.0016271, 0.0014962
6: 0.0036712, 0.0045510, 0.0036449, 0.0045047, -0.0007278, 0.0007914
7: -0.0126569, -0.0065582, -0.0123360, -0.0063764, -0.0054856, 0.0050446
8: 0.0066878, 0.0115262, 0.0069424, 0.0116704, -0.0043520, 0.0040021
9: 0.0097532, 0.0184556, 0.0102111, 0.0187149, -0.0078275, 0.0071982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019061, upper bound: 0.0020046
time: 1.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019061, upper bound: 0.0019953
time: 1.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041700, -0.0040973, -0.0041706, -0.0041006, -0.0000601, 0.0000638
1: -0.0088234, -0.0061005, -0.0088450, -0.0062246, -0.0022509, 0.0023879
2: 0.9658750, 0.9691426, 0.9658492, 0.9689937, -0.0027011, 0.0028656
3: -0.0053939, 0.0187067, -0.0055850, 0.0176087, -0.0199230, 0.0211362
4: -0.0021158, -0.0002828, -0.0020323, -0.0002683, -0.0016075, 0.0015153
5: 0.0151320, 0.0169845, 0.0152164, 0.0169992, -0.0016247, 0.0015314
6: 0.0036455, 0.0045466, 0.0036384, 0.0045055, -0.0007449, 0.0007902
7: -0.0126263, -0.0063804, -0.0123417, -0.0063308, -0.0054776, 0.0051632
8: 0.0067121, 0.0116673, 0.0069378, 0.0117065, -0.0043457, 0.0040963
9: 0.0097969, 0.0187093, 0.0102030, 0.0187800, -0.0078161, 0.0073675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020170, upper bound: 0.0019837
time: 1.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020170, upper bound: 0.0020528
time: 1.27 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041680, -0.0040970, -0.0041703, -0.0040992, -0.0000602, 0.0000644
1: -0.0087459, -0.0060872, -0.0088338, -0.0061727, -0.0022562, 0.0024118
2: 0.9659681, 0.9691586, 0.9658625, 0.9690560, -0.0027075, 0.0028942
3: -0.0047079, 0.0188249, -0.0054865, 0.0180678, -0.0199704, 0.0213472
4: -0.0021248, -0.0003350, -0.0020672, -0.0002758, -0.0016236, 0.0015189
5: 0.0151229, 0.0169318, 0.0151811, 0.0169916, -0.0016409, 0.0015351
6: 0.0036712, 0.0045510, 0.0036421, 0.0045227, -0.0007467, 0.0007981
7: -0.0126569, -0.0065582, -0.0124607, -0.0063564, -0.0055323, 0.0051755
8: 0.0066878, 0.0115262, 0.0068434, 0.0116863, -0.0043891, 0.0041060
9: 0.0097532, 0.0184556, 0.0100332, 0.0187435, -0.0078942, 0.0073850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018746, upper bound: 0.0019960
time: 1.45 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018937, upper bound: 0.0020168
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041700, -0.0040973, -0.0041708, -0.0040992, -0.0000615, 0.0000643
1: -0.0088234, -0.0061005, -0.0088538, -0.0061703, -0.0023030, 0.0024086
2: 0.9658750, 0.9691426, 0.9658386, 0.9690588, -0.0027637, 0.0028905
3: -0.0053939, 0.0187067, -0.0056637, 0.0180892, -0.0203849, 0.0213195
4: -0.0021158, -0.0002828, -0.0020688, -0.0002623, -0.0016215, 0.0015504
5: 0.0151320, 0.0169845, 0.0151794, 0.0170053, -0.0016388, 0.0015669
6: 0.0036455, 0.0045466, 0.0036354, 0.0045235, -0.0007622, 0.0007971
7: -0.0126263, -0.0063804, -0.0124662, -0.0063104, -0.0055251, 0.0052829
8: 0.0067121, 0.0116673, 0.0068390, 0.0117227, -0.0043834, 0.0041912
9: 0.0097969, 0.0187093, 0.0100253, 0.0188091, -0.0078839, 0.0075383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019824, upper bound: 0.0019792
time: 1.31 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019824, upper bound: 0.0020703
time: 1.79 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041683, -0.0040954, -0.0041714, -0.0041003, -0.0000597, 0.0000668
1: -0.0087586, -0.0060303, -0.0088762, -0.0062109, -0.0022357, 0.0025020
2: 0.9659528, 0.9692270, 0.9658116, 0.9690102, -0.0026830, 0.0030025
3: -0.0048207, 0.0193286, -0.0058616, 0.0177299, -0.0197890, 0.0221457
4: -0.0021631, -0.0003264, -0.0020415, -0.0002472, -0.0016843, 0.0015051
5: 0.0150842, 0.0169405, 0.0152071, 0.0170205, -0.0017023, 0.0015211
6: 0.0036669, 0.0045698, 0.0036280, 0.0045101, -0.0007399, 0.0008280
7: -0.0127874, -0.0065289, -0.0123731, -0.0062592, -0.0057392, 0.0051285
8: 0.0065842, 0.0115494, 0.0069129, 0.0117634, -0.0045532, 0.0040687
9: 0.0095669, 0.0184973, 0.0101581, 0.0188822, -0.0081894, 0.0073179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019513, upper bound: 0.0020165
time: 1.28 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019513, upper bound: 0.0020165
time: 1.31 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041704, -0.0040957, -0.0041714, -0.0041003, -0.0000611, 0.0000655
1: -0.0088385, -0.0060417, -0.0088762, -0.0062109, -0.0022862, 0.0024513
2: 0.9658570, 0.9692132, 0.9658116, 0.9690102, -0.0027436, 0.0029417
3: -0.0055278, 0.0192278, -0.0058616, 0.0177299, -0.0202363, 0.0216972
4: -0.0021554, -0.0002726, -0.0020415, -0.0002472, -0.0016502, 0.0015391
5: 0.0150919, 0.0169948, 0.0152071, 0.0170205, -0.0016678, 0.0015555
6: 0.0036405, 0.0045661, 0.0036280, 0.0045101, -0.0007566, 0.0008112
7: -0.0127613, -0.0063457, -0.0123731, -0.0062592, -0.0056230, 0.0052444
8: 0.0066049, 0.0116948, 0.0069129, 0.0117634, -0.0044610, 0.0041607
9: 0.0096042, 0.0187588, 0.0101581, 0.0188822, -0.0080236, 0.0074833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019513, upper bound: 0.0019814
time: 1.82 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019513, upper bound: 0.0019814
time: 1.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041680, -0.0040970, -0.0041701, -0.0040973, -0.0000582, 0.0000610
1: -0.0087459, -0.0060872, -0.0088279, -0.0060994, -0.0021802, 0.0022857
2: 0.9659681, 0.9691586, 0.9658697, 0.9691439, -0.0026164, 0.0027429
3: -0.0047079, 0.0188249, -0.0054340, 0.0187167, -0.0192980, 0.0202314
4: -0.0021248, -0.0003350, -0.0021165, -0.0002797, -0.0015387, 0.0014677
5: 0.0151229, 0.0169318, 0.0151312, 0.0169876, -0.0015551, 0.0014834
6: 0.0036712, 0.0045510, 0.0036440, 0.0045470, -0.0007215, 0.0007564
7: -0.0126569, -0.0065582, -0.0126289, -0.0063700, -0.0052431, 0.0050012
8: 0.0066878, 0.0115262, 0.0067100, 0.0116755, -0.0041597, 0.0039677
9: 0.0097532, 0.0184556, 0.0097932, 0.0187241, -0.0074815, 0.0071364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019043, upper bound: 0.0020046
time: 1.25 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019043, upper bound: 0.0019953
time: 1.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041700, -0.0040973, -0.0041707, -0.0040972, -0.0000597, 0.0000608
1: -0.0088234, -0.0061005, -0.0088481, -0.0060969, -0.0022342, 0.0022768
2: 0.9658750, 0.9691426, 0.9658453, 0.9691469, -0.0026811, 0.0027322
3: -0.0053939, 0.0187067, -0.0056128, 0.0187389, -0.0197752, 0.0201525
4: -0.0021158, -0.0002828, -0.0021182, -0.0002661, -0.0015327, 0.0015040
5: 0.0151320, 0.0169845, 0.0151295, 0.0170014, -0.0015491, 0.0015201
6: 0.0036455, 0.0045466, 0.0036373, 0.0045478, -0.0007394, 0.0007535
7: -0.0126263, -0.0063804, -0.0126346, -0.0063237, -0.0052227, 0.0051249
8: 0.0067121, 0.0116673, 0.0067055, 0.0117123, -0.0041434, 0.0040659
9: 0.0097969, 0.0187093, 0.0097850, 0.0187902, -0.0074524, 0.0073128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020166, upper bound: 0.0019836
time: 1.34 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020166, upper bound: 0.0020529
time: 1.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041680, -0.0040970, -0.0041705, -0.0040957, -0.0000597, 0.0000616
1: -0.0087459, -0.0060872, -0.0088417, -0.0060402, -0.0022366, 0.0023054
2: 0.9659681, 0.9691586, 0.9658530, 0.9692150, -0.0026840, 0.0027666
3: -0.0047079, 0.0188249, -0.0055564, 0.0192405, -0.0197965, 0.0204061
4: -0.0021248, -0.0003350, -0.0021564, -0.0002704, -0.0015520, 0.0015056
5: 0.0151229, 0.0169318, 0.0150909, 0.0169970, -0.0015686, 0.0015217
6: 0.0036712, 0.0045510, 0.0036394, 0.0045666, -0.0007402, 0.0007629
7: -0.0126569, -0.0065582, -0.0127646, -0.0063383, -0.0052884, 0.0051304
8: 0.0066878, 0.0115262, 0.0066023, 0.0117007, -0.0041956, 0.0040702
9: 0.0097532, 0.0184556, 0.0095995, 0.0187694, -0.0075461, 0.0073207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018735, upper bound: 0.0019960
time: 1.46 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018913, upper bound: 0.0020168
time: 1.30 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041700, -0.0040973, -0.0041711, -0.0040956, -0.0000610, 0.0000613
1: -0.0088234, -0.0061005, -0.0088624, -0.0060377, -0.0022860, 0.0022969
2: 0.9658750, 0.9691426, 0.9658282, 0.9692180, -0.0027433, 0.0027564
3: -0.0053939, 0.0187067, -0.0057391, 0.0192624, -0.0202343, 0.0203306
4: -0.0021158, -0.0002828, -0.0021580, -0.0002565, -0.0015463, 0.0015389
5: 0.0151320, 0.0169845, 0.0150893, 0.0170111, -0.0015628, 0.0015554
6: 0.0036455, 0.0045466, 0.0036326, 0.0045674, -0.0007565, 0.0007601
7: -0.0126263, -0.0063804, -0.0127703, -0.0062909, -0.0052688, 0.0052439
8: 0.0067121, 0.0116673, 0.0065978, 0.0117382, -0.0041801, 0.0041603
9: 0.0097969, 0.0187093, 0.0095914, 0.0188370, -0.0075182, 0.0074826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019824, upper bound: 0.0019792
time: 1.33 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019824, upper bound: 0.0020703
time: 1.25 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041683, -0.0040954, -0.0041716, -0.0040969, -0.0000592, 0.0000639
1: -0.0087586, -0.0060303, -0.0088821, -0.0060833, -0.0022180, 0.0023918
2: 0.9659528, 0.9692270, 0.9658045, 0.9691633, -0.0026617, 0.0028702
3: -0.0048207, 0.0193286, -0.0059137, 0.0188597, -0.0196324, 0.0211704
4: -0.0021631, -0.0003264, -0.0021274, -0.0002433, -0.0016101, 0.0014932
5: 0.0150842, 0.0169405, 0.0151202, 0.0170245, -0.0016273, 0.0015091
6: 0.0036669, 0.0045698, 0.0036261, 0.0045523, -0.0007340, 0.0007915
7: -0.0127874, -0.0065289, -0.0126659, -0.0062457, -0.0054865, 0.0050879
8: 0.0065842, 0.0115494, 0.0066806, 0.0117741, -0.0043527, 0.0040365
9: 0.0095669, 0.0184973, 0.0097403, 0.0189015, -0.0078288, 0.0072600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019504, upper bound: 0.0020165
time: 1.32 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019504, upper bound: 0.0020165
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041704, -0.0040957, -0.0041716, -0.0040969, -0.0000606, 0.0000625
1: -0.0088385, -0.0060417, -0.0088821, -0.0060833, -0.0022699, 0.0023418
2: 0.9658570, 0.9692132, 0.9658045, 0.9691633, -0.0027240, 0.0028103
3: -0.0055278, 0.0192278, -0.0059137, 0.0188597, -0.0200915, 0.0207280
4: -0.0021554, -0.0002726, -0.0021274, -0.0002433, -0.0015765, 0.0015281
5: 0.0150919, 0.0169948, 0.0151202, 0.0170245, -0.0015933, 0.0015444
6: 0.0036405, 0.0045661, 0.0036261, 0.0045523, -0.0007512, 0.0007750
7: -0.0127613, -0.0063457, -0.0126659, -0.0062457, -0.0053718, 0.0052069
8: 0.0066049, 0.0116948, 0.0066806, 0.0117741, -0.0042618, 0.0041309
9: 0.0096042, 0.0187588, 0.0097403, 0.0189015, -0.0076652, 0.0074298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019504, upper bound: 0.0019815
time: 1.86 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019504, upper bound: 0.0019814
time: 1.77 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 5.14 seconds
IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0019061, upper bound: 0.0020046
IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0019061, upper bound: 0.0019953
IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0020170, upper bound: 0.0019837
IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0020170, upper bound: 0.0020528
IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0018746, upper bound: 0.0019960
IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0018937, upper bound: 0.0020168
IS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0019824, upper bound: 0.0019792
IS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0019824, upper bound: 0.0020703
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0019513, upper bound: 0.0020165
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0019513, upper bound: 0.0020165
IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0019513, upper bound: 0.0019814
IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0019513, upper bound: 0.0019814
IS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0019043, upper bound: 0.0020046
IS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0019043, upper bound: 0.0019953
IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0020166, upper bound: 0.0019836
IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0020166, upper bound: 0.0020529
IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0018735, upper bound: 0.0019960
IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0018913, upper bound: 0.0020168
IS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0019824, upper bound: 0.0019792
IS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0019824, upper bound: 0.0020703
IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0019504, upper bound: 0.0020165
IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0019504, upper bound: 0.0020165
IS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0019504, upper bound: 0.0019815
IS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 5.14
Output dim: 2, lower bound: -0.0019504, upper bound: 0.0019814

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041680, -0.0040970, -0.0041691, -0.0041017, -0.0000576, 0.0000626
1: -0.0087459, -0.0060872, -0.0087867, -0.0062662, -0.0021582, 0.0023430
2: 0.9659681, 0.9691586, 0.9659190, 0.9689438, -0.0025899, 0.0028117
3: -0.0047079, 0.0188249, -0.0050698, 0.0172399, -0.0191026, 0.0207386
4: -0.0021248, -0.0003350, -0.0020042, -0.0003074, -0.0015773, 0.0014529
5: 0.0151229, 0.0169318, 0.0152447, 0.0169596, -0.0015941, 0.0014684
6: 0.0036712, 0.0045510, 0.0036576, 0.0044918, -0.0007142, 0.0007754
7: -0.0126569, -0.0065582, -0.0122461, -0.0064644, -0.0053746, 0.0049506
8: 0.0066878, 0.0115262, 0.0070136, 0.0116006, -0.0042639, 0.0039276
9: 0.0097532, 0.0184556, 0.0103393, 0.0185894, -0.0076691, 0.0070641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018441, upper bound: 0.0019542
time: 1.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018667, upper bound: 0.0019644
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041700, -0.0040973, -0.0041679, -0.0041003, -0.0000614, 0.0000612
1: -0.0088234, -0.0061005, -0.0087450, -0.0062114, -0.0022978, 0.0022904
2: 0.9658750, 0.9691426, 0.9659691, 0.9690095, -0.0027575, 0.0027486
3: -0.0053939, 0.0187067, -0.0047004, 0.0177253, -0.0203389, 0.0202729
4: -0.0021158, -0.0002828, -0.0020411, -0.0003355, -0.0015419, 0.0015469
5: 0.0151320, 0.0169845, 0.0152074, 0.0169312, -0.0015583, 0.0015634
6: 0.0036455, 0.0045466, 0.0036714, 0.0045099, -0.0007604, 0.0007580
7: -0.0126263, -0.0063804, -0.0123719, -0.0065601, -0.0052539, 0.0052710
8: 0.0067121, 0.0116673, 0.0069139, 0.0115247, -0.0041682, 0.0041818
9: 0.0097969, 0.0187093, 0.0101598, 0.0184528, -0.0074969, 0.0075213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019691, upper bound: 0.0019308
time: 1.33 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019653, upper bound: 0.0019308
time: 1.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041700, -0.0040973, -0.0041699, -0.0041007, -0.0000600, 0.0000624
1: -0.0088234, -0.0061005, -0.0088193, -0.0062280, -0.0022475, 0.0023363
2: 0.9658750, 0.9691426, 0.9658799, 0.9689896, -0.0026971, 0.0028037
3: -0.0053939, 0.0187067, -0.0053576, 0.0175783, -0.0198936, 0.0206797
4: -0.0021158, -0.0002828, -0.0020300, -0.0002856, -0.0015728, 0.0015130
5: 0.0151320, 0.0169845, 0.0152187, 0.0169817, -0.0015896, 0.0015292
6: 0.0036455, 0.0045466, 0.0036469, 0.0045044, -0.0007438, 0.0007732
7: -0.0126263, -0.0063804, -0.0123338, -0.0063898, -0.0053593, 0.0051556
8: 0.0067121, 0.0116673, 0.0069441, 0.0116598, -0.0042518, 0.0040902
9: 0.0097969, 0.0187093, 0.0102142, 0.0186959, -0.0076473, 0.0073567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019691, upper bound: 0.0019468
time: 1.84 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019653, upper bound: 0.0019468
time: 1.96 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041674, -0.0040973, -0.0041682, -0.0041001, -0.0000581, 0.0000614
1: -0.0087265, -0.0061005, -0.0087534, -0.0062031, -0.0021760, 0.0022981
2: 0.9659913, 0.9691426, 0.9659591, 0.9690195, -0.0026114, 0.0027578
3: -0.0045363, 0.0187070, -0.0047747, 0.0177987, -0.0192609, 0.0203409
4: -0.0021158, -0.0003480, -0.0020467, -0.0003299, -0.0015470, 0.0014649
5: 0.0151319, 0.0169186, 0.0152018, 0.0169369, -0.0015636, 0.0014805
6: 0.0036776, 0.0045466, 0.0036687, 0.0045126, -0.0007201, 0.0007605
7: -0.0126263, -0.0066026, -0.0123909, -0.0065408, -0.0052715, 0.0049916
8: 0.0067120, 0.0114909, 0.0068988, 0.0115399, -0.0041822, 0.0039601
9: 0.0097968, 0.0183922, 0.0101327, 0.0184803, -0.0075221, 0.0071226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018187, upper bound: 0.0019406
time: 1.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018121, upper bound: 0.0019406
time: 1.43 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041680, -0.0040970, -0.0041692, -0.0040998, -0.0000590, 0.0000637
1: -0.0087459, -0.0060872, -0.0087905, -0.0061937, -0.0022107, 0.0023848
2: 0.9659681, 0.9691586, 0.9659145, 0.9690308, -0.0026529, 0.0028618
3: -0.0047079, 0.0188249, -0.0051027, 0.0178820, -0.0195674, 0.0211082
4: -0.0021248, -0.0003350, -0.0020531, -0.0003049, -0.0016054, 0.0014882
5: 0.0151229, 0.0169318, 0.0151954, 0.0169621, -0.0016225, 0.0015041
6: 0.0036712, 0.0045510, 0.0036564, 0.0045158, -0.0007316, 0.0007892
7: -0.0126569, -0.0065582, -0.0124125, -0.0064558, -0.0054704, 0.0050711
8: 0.0066878, 0.0115262, 0.0068816, 0.0116074, -0.0043399, 0.0040231
9: 0.0097532, 0.0184556, 0.0101019, 0.0186016, -0.0078058, 0.0072360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018643, upper bound: 0.0020024
time: 1.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018643, upper bound: 0.0020168
time: 1.96 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041700, -0.0040973, -0.0041702, -0.0040993, -0.0000614, 0.0000629
1: -0.0088234, -0.0061005, -0.0088284, -0.0061740, -0.0022995, 0.0023554
2: 0.9658750, 0.9691426, 0.9658690, 0.9690545, -0.0027595, 0.0028266
3: -0.0053939, 0.0187067, -0.0054388, 0.0180568, -0.0203536, 0.0208482
4: -0.0021158, -0.0002828, -0.0020664, -0.0002794, -0.0015856, 0.0015480
5: 0.0151320, 0.0169845, 0.0151819, 0.0169880, -0.0016026, 0.0015645
6: 0.0036455, 0.0045466, 0.0036438, 0.0045223, -0.0007610, 0.0007795
7: -0.0126263, -0.0063804, -0.0124578, -0.0063687, -0.0054030, 0.0052748
8: 0.0067121, 0.0116673, 0.0068457, 0.0116765, -0.0042865, 0.0041848
9: 0.0097969, 0.0187093, 0.0100372, 0.0187259, -0.0077097, 0.0075267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019313, upper bound: 0.0019546
time: 1.32 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019274, upper bound: 0.0019546
time: 2.05 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041683, -0.0040954, -0.0041699, -0.0041007, -0.0000591, 0.0000652
1: -0.0087586, -0.0060303, -0.0088193, -0.0062280, -0.0022145, 0.0024400
2: 0.9659528, 0.9692270, 0.9658799, 0.9689896, -0.0026575, 0.0029281
3: -0.0048207, 0.0193286, -0.0053576, 0.0175783, -0.0196015, 0.0215971
4: -0.0021631, -0.0003264, -0.0020300, -0.0002856, -0.0016426, 0.0014908
5: 0.0150842, 0.0169405, 0.0152187, 0.0169817, -0.0016601, 0.0015067
6: 0.0036669, 0.0045698, 0.0036469, 0.0045044, -0.0007329, 0.0008075
7: -0.0127874, -0.0065289, -0.0123338, -0.0063898, -0.0055971, 0.0050799
8: 0.0065842, 0.0115494, 0.0069441, 0.0116598, -0.0044404, 0.0040302
9: 0.0095669, 0.0184973, 0.0102142, 0.0186959, -0.0079866, 0.0072486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018813, upper bound: 0.0019655
time: 2.01 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019118, upper bound: 0.0019772
time: 1.30 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041683, -0.0040954, -0.0041702, -0.0040993, -0.0000597, 0.0000648
1: -0.0087586, -0.0060303, -0.0088284, -0.0061740, -0.0022354, 0.0024266
2: 0.9659528, 0.9692270, 0.9658690, 0.9690545, -0.0026826, 0.0029121
3: -0.0048207, 0.0193286, -0.0054388, 0.0180568, -0.0197864, 0.0214789
4: -0.0021631, -0.0003264, -0.0020664, -0.0002794, -0.0016336, 0.0015049
5: 0.0150842, 0.0169405, 0.0151819, 0.0169880, -0.0016510, 0.0015209
6: 0.0036669, 0.0045698, 0.0036438, 0.0045223, -0.0007398, 0.0008031
7: -0.0127874, -0.0065289, -0.0124578, -0.0063687, -0.0055664, 0.0051278
8: 0.0065842, 0.0115494, 0.0068457, 0.0116765, -0.0044162, 0.0040682
9: 0.0095669, 0.0184973, 0.0100372, 0.0187259, -0.0079429, 0.0073170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018813, upper bound: 0.0019655
time: 1.47 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019118, upper bound: 0.0019772
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041680, -0.0040970, -0.0041692, -0.0040984, -0.0000571, 0.0000597
1: -0.0087459, -0.0060872, -0.0087922, -0.0061409, -0.0021392, 0.0022374
2: 0.9659681, 0.9691586, 0.9659125, 0.9690940, -0.0025671, 0.0026850
3: -0.0047079, 0.0188249, -0.0051181, 0.0183492, -0.0189348, 0.0198039
4: -0.0021248, -0.0003350, -0.0020886, -0.0003038, -0.0015062, 0.0014401
5: 0.0151229, 0.0169318, 0.0151595, 0.0169633, -0.0015223, 0.0014555
6: 0.0036712, 0.0045510, 0.0036558, 0.0045332, -0.0007079, 0.0007404
7: -0.0126569, -0.0065582, -0.0125336, -0.0064519, -0.0051324, 0.0049071
8: 0.0066878, 0.0115262, 0.0067856, 0.0116106, -0.0040718, 0.0038931
9: 0.0097532, 0.0184556, 0.0099291, 0.0186073, -0.0073235, 0.0070021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018445, upper bound: 0.0019542
time: 1.25 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018647, upper bound: 0.0019644
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041700, -0.0040973, -0.0041680, -0.0040970, -0.0000609, 0.0000582
1: -0.0088234, -0.0061005, -0.0087459, -0.0060872, -0.0022791, 0.0021793
2: 0.9658750, 0.9691426, 0.9659681, 0.9691586, -0.0027350, 0.0026153
3: -0.0053939, 0.0187067, -0.0047079, 0.0188249, -0.0201727, 0.0192896
4: -0.0021158, -0.0002828, -0.0021248, -0.0003350, -0.0014671, 0.0015343
5: 0.0151320, 0.0169845, 0.0151229, 0.0169318, -0.0014828, 0.0015506
6: 0.0036455, 0.0045466, 0.0036712, 0.0045510, -0.0007542, 0.0007212
7: -0.0126263, -0.0063804, -0.0126569, -0.0065582, -0.0049991, 0.0052279
8: 0.0067121, 0.0116673, 0.0066878, 0.0115262, -0.0039660, 0.0041476
9: 0.0097969, 0.0187093, 0.0097532, 0.0184556, -0.0071333, 0.0074599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019689, upper bound: 0.0019308
time: 1.38 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019651, upper bound: 0.0019308
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041700, -0.0040973, -0.0041700, -0.0040973, -0.0000596, 0.0000596
1: -0.0088234, -0.0061005, -0.0088234, -0.0061005, -0.0022307, 0.0022307
2: 0.9658750, 0.9691426, 0.9658750, 0.9691426, -0.0026770, 0.0026770
3: -0.0053939, 0.0187067, -0.0053939, 0.0187067, -0.0197449, 0.0197449
4: -0.0021158, -0.0002828, -0.0021158, -0.0002828, -0.0015017, 0.0015017
5: 0.0151320, 0.0169845, 0.0151320, 0.0169845, -0.0015178, 0.0015178
6: 0.0036455, 0.0045466, 0.0036455, 0.0045466, -0.0007382, 0.0007382
7: -0.0126263, -0.0063804, -0.0126263, -0.0063804, -0.0051171, 0.0051171
8: 0.0067121, 0.0116673, 0.0067121, 0.0116673, -0.0040596, 0.0040596
9: 0.0097969, 0.0187093, 0.0097969, 0.0187093, -0.0073017, 0.0073017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019689, upper bound: 0.0019468
time: 1.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019651, upper bound: 0.0019468
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041674, -0.0040973, -0.0041683, -0.0040967, -0.0000575, 0.0000586
1: -0.0087265, -0.0061005, -0.0087603, -0.0060772, -0.0021527, 0.0021941
2: 0.9659913, 0.9691426, 0.9659508, 0.9691706, -0.0025833, 0.0026330
3: -0.0045363, 0.0187070, -0.0048353, 0.0189136, -0.0190542, 0.0194203
4: -0.0021158, -0.0003480, -0.0021315, -0.0003253, -0.0014770, 0.0014492
5: 0.0151319, 0.0169186, 0.0151161, 0.0169416, -0.0014928, 0.0014647
6: 0.0036776, 0.0045466, 0.0036664, 0.0045543, -0.0007124, 0.0007261
7: -0.0126263, -0.0066026, -0.0126799, -0.0065251, -0.0050329, 0.0049381
8: 0.0067120, 0.0114909, 0.0066695, 0.0115524, -0.0039929, 0.0039176
9: 0.0097968, 0.0183922, 0.0097204, 0.0185027, -0.0071816, 0.0070462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018239, upper bound: 0.0019408
time: 1.27 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018195, upper bound: 0.0019408
time: 1.52 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041680, -0.0040970, -0.0041694, -0.0040963, -0.0000585, 0.0000605
1: -0.0087459, -0.0060872, -0.0087999, -0.0060608, -0.0021906, 0.0022668
2: 0.9659681, 0.9691586, 0.9659032, 0.9691902, -0.0026288, 0.0027202
3: -0.0047079, 0.0188249, -0.0051861, 0.0190587, -0.0193897, 0.0200638
4: -0.0021248, -0.0003350, -0.0021426, -0.0002986, -0.0015260, 0.0014747
5: 0.0151229, 0.0169318, 0.0151049, 0.0169686, -0.0015423, 0.0014904
6: 0.0036712, 0.0045510, 0.0036533, 0.0045598, -0.0007250, 0.0007502
7: -0.0126569, -0.0065582, -0.0127175, -0.0064342, -0.0051997, 0.0050250
8: 0.0066878, 0.0115262, 0.0066397, 0.0116245, -0.0041252, 0.0039866
9: 0.0097532, 0.0184556, 0.0096668, 0.0186324, -0.0074196, 0.0071703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018641, upper bound: 0.0020024
time: 1.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018641, upper bound: 0.0020168
time: 2.00 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041700, -0.0040973, -0.0041704, -0.0040957, -0.0000609, 0.0000601
1: -0.0088234, -0.0061005, -0.0088385, -0.0060417, -0.0022824, 0.0022492
2: 0.9658750, 0.9691426, 0.9658570, 0.9692132, -0.0027390, 0.0026991
3: -0.0053939, 0.0187067, -0.0055278, 0.0192278, -0.0202021, 0.0199083
4: -0.0021158, -0.0002828, -0.0021554, -0.0002726, -0.0015141, 0.0015365
5: 0.0151320, 0.0169845, 0.0150919, 0.0169948, -0.0015303, 0.0015529
6: 0.0036455, 0.0045466, 0.0036405, 0.0045661, -0.0007553, 0.0007443
7: -0.0126263, -0.0063804, -0.0127613, -0.0063457, -0.0051594, 0.0052355
8: 0.0067121, 0.0116673, 0.0066049, 0.0116948, -0.0040932, 0.0041536
9: 0.0097969, 0.0187093, 0.0096042, 0.0187588, -0.0073621, 0.0074707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019320, upper bound: 0.0019546
time: 1.28 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019288, upper bound: 0.0019546
time: 1.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041683, -0.0040954, -0.0041700, -0.0040973, -0.0000587, 0.0000622
1: -0.0087586, -0.0060303, -0.0088234, -0.0061005, -0.0021966, 0.0023306
2: 0.9659528, 0.9692270, 0.9658750, 0.9691426, -0.0026360, 0.0027968
3: -0.0048207, 0.0193286, -0.0053939, 0.0187067, -0.0194430, 0.0206290
4: -0.0021631, -0.0003264, -0.0021158, -0.0002828, -0.0015690, 0.0014788
5: 0.0150842, 0.0169405, 0.0151320, 0.0169845, -0.0015857, 0.0014945
6: 0.0036669, 0.0045698, 0.0036455, 0.0045466, -0.0007269, 0.0007713
7: -0.0127874, -0.0065289, -0.0126263, -0.0063804, -0.0053462, 0.0050388
8: 0.0065842, 0.0115494, 0.0067121, 0.0116673, -0.0042414, 0.0039976
9: 0.0095669, 0.0184973, 0.0097969, 0.0187093, -0.0076286, 0.0071900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018831, upper bound: 0.0019655
time: 1.44 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019112, upper bound: 0.0019772
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041683, -0.0040954, -0.0041704, -0.0040957, -0.0000592, 0.0000620
1: -0.0087586, -0.0060303, -0.0088385, -0.0060417, -0.0022183, 0.0023204
2: 0.9659528, 0.9692270, 0.9658570, 0.9692132, -0.0026620, 0.0027846
3: -0.0048207, 0.0193286, -0.0055278, 0.0192278, -0.0196344, 0.0205389
4: -0.0021631, -0.0003264, -0.0021554, -0.0002726, -0.0015621, 0.0014933
5: 0.0150842, 0.0169405, 0.0150919, 0.0169948, -0.0015788, 0.0015093
6: 0.0036669, 0.0045698, 0.0036405, 0.0045661, -0.0007341, 0.0007679
7: -0.0127874, -0.0065289, -0.0127613, -0.0063457, -0.0053228, 0.0050884
8: 0.0065842, 0.0115494, 0.0066049, 0.0116948, -0.0042229, 0.0040369
9: 0.0095669, 0.0184973, 0.0096042, 0.0187588, -0.0075953, 0.0072608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018831, upper bound: 0.0019655
time: 1.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019112, upper bound: 0.0019772
time: 1.36 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 4.62 seconds
IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0018441, upper bound: 0.0019542
IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0018667, upper bound: 0.0019644
IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0019691, upper bound: 0.0019308
IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0019653, upper bound: 0.0019308
IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0019691, upper bound: 0.0019468
IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0019653, upper bound: 0.0019468
IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0018187, upper bound: 0.0019406
IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0018121, upper bound: 0.0019406
IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0018643, upper bound: 0.0020024
IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0018643, upper bound: 0.0020168
IS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0019313, upper bound: 0.0019546
IS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0019274, upper bound: 0.0019546
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0018813, upper bound: 0.0019655
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0019118, upper bound: 0.0019772
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0018813, upper bound: 0.0019655
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0019118, upper bound: 0.0019772
IS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0018445, upper bound: 0.0019542
IS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0018647, upper bound: 0.0019644
IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0019689, upper bound: 0.0019308
IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0019651, upper bound: 0.0019308
IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0019689, upper bound: 0.0019468
IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0019651, upper bound: 0.0019468
IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0018239, upper bound: 0.0019408
IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0018195, upper bound: 0.0019408
IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0018641, upper bound: 0.0020024
IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0018641, upper bound: 0.0020168
IS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0019320, upper bound: 0.0019546
IS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0019288, upper bound: 0.0019546
IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0018831, upper bound: 0.0019655
IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0019112, upper bound: 0.0019772
IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0018831, upper bound: 0.0019655
IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 11, time: 4.62
Output dim: 2, lower bound: -0.0019112, upper bound: 0.0019772

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041656, -0.0040980, -0.0041692, -0.0040998, -0.0000564, 0.0000613
1: -0.0086578, -0.0061267, -0.0087905, -0.0061937, -0.0021118, 0.0022954
2: 0.9660737, 0.9691111, 0.9659145, 0.9690308, -0.0025343, 0.0027546
3: -0.0039287, 0.0184748, -0.0051027, 0.0178820, -0.0186923, 0.0203174
4: -0.0020981, -0.0003942, -0.0020531, -0.0003049, -0.0015453, 0.0014217
5: 0.0151498, 0.0168719, 0.0151954, 0.0169621, -0.0015618, 0.0014368
6: 0.0037003, 0.0045379, 0.0036564, 0.0045158, -0.0006989, 0.0007596
7: -0.0125662, -0.0067601, -0.0124125, -0.0064558, -0.0052654, 0.0048443
8: 0.0067597, 0.0113660, 0.0068816, 0.0116074, -0.0041773, 0.0038432
9: 0.0098827, 0.0181675, 0.0101019, 0.0186016, -0.0075134, 0.0069124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018031, upper bound: 0.0019577
time: 2.05 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018019, upper bound: 0.0019456
time: 1.96 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041669, -0.0040975, -0.0041692, -0.0040998, -0.0000583, 0.0000626
1: -0.0087066, -0.0061088, -0.0087905, -0.0061937, -0.0021842, 0.0023437
2: 0.9660152, 0.9691326, 0.9659145, 0.9690308, -0.0026211, 0.0028125
3: -0.0043600, 0.0186331, -0.0051027, 0.0178820, -0.0193327, 0.0207448
4: -0.0021102, -0.0003614, -0.0020531, -0.0003049, -0.0015778, 0.0014704
5: 0.0151376, 0.0169051, 0.0151954, 0.0169621, -0.0015946, 0.0014861
6: 0.0036842, 0.0045438, 0.0036564, 0.0045158, -0.0007228, 0.0007756
7: -0.0126072, -0.0066483, -0.0124125, -0.0064558, -0.0053762, 0.0050102
8: 0.0067272, 0.0114547, 0.0068816, 0.0116074, -0.0042652, 0.0039749
9: 0.0098241, 0.0183270, 0.0101019, 0.0186016, -0.0076714, 0.0071492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018031, upper bound: 0.0019677
time: 1.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018019, upper bound: 0.0019563
time: 1.47 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041656, -0.0040980, -0.0041694, -0.0040963, -0.0000559, 0.0000586
1: -0.0086578, -0.0061267, -0.0087999, -0.0060608, -0.0020937, 0.0021952
2: 0.9660737, 0.9691111, 0.9659032, 0.9691902, -0.0025125, 0.0026343
3: -0.0039287, 0.0184748, -0.0051861, 0.0190587, -0.0185318, 0.0194304
4: -0.0020981, -0.0003942, -0.0021426, -0.0002986, -0.0014778, 0.0014094
5: 0.0151498, 0.0168719, 0.0151049, 0.0169686, -0.0014936, 0.0014245
6: 0.0037003, 0.0045379, 0.0036533, 0.0045598, -0.0006929, 0.0007265
7: -0.0125662, -0.0067601, -0.0127175, -0.0064342, -0.0050356, 0.0048027
8: 0.0067597, 0.0113660, 0.0066397, 0.0116245, -0.0039950, 0.0038102
9: 0.0098827, 0.0181675, 0.0096668, 0.0186324, -0.0071853, 0.0068530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018104, upper bound: 0.0019577
time: 1.46 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018104, upper bound: 0.0019458
time: 2.02 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041669, -0.0040975, -0.0041694, -0.0040963, -0.0000577, 0.0000594
1: -0.0087066, -0.0061088, -0.0087999, -0.0060608, -0.0021624, 0.0022258
2: 0.9660152, 0.9691326, 0.9659032, 0.9691902, -0.0025950, 0.0026711
3: -0.0043600, 0.0186331, -0.0051861, 0.0190587, -0.0191402, 0.0197017
4: -0.0021102, -0.0003614, -0.0021426, -0.0002986, -0.0014984, 0.0014557
5: 0.0151376, 0.0169051, 0.0151049, 0.0169686, -0.0015144, 0.0014713
6: 0.0036842, 0.0045438, 0.0036533, 0.0045598, -0.0007156, 0.0007366
7: -0.0126072, -0.0066483, -0.0127175, -0.0064342, -0.0051059, 0.0049603
8: 0.0067272, 0.0114547, 0.0066397, 0.0116245, -0.0040507, 0.0039353
9: 0.0098241, 0.0183270, 0.0096668, 0.0186324, -0.0072857, 0.0070780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018104, upper bound: 0.0019677
time: 1.89 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018104, upper bound: 0.0019563
time: 2.02 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 5.42 seconds
IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 5.42
Output dim: 2, lower bound: -0.0018031, upper bound: 0.0019577
IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 12, time: 5.42
Output dim: 2, lower bound: -0.0018019, upper bound: 0.0019456
IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 5.42
Output dim: 2, lower bound: -0.0018031, upper bound: 0.0019677
IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 12, time: 5.42
Output dim: 2, lower bound: -0.0018019, upper bound: 0.0019563
IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 5.42
Output dim: 2, lower bound: -0.0018104, upper bound: 0.0019577
IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 12, time: 5.42
Output dim: 2, lower bound: -0.0018104, upper bound: 0.0019458
IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 5.42
Output dim: 2, lower bound: -0.0018104, upper bound: 0.0019677
IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 12, time: 5.42
Output dim: 2, lower bound: -0.0018104, upper bound: 0.0019563

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.91 + 386.32 = 390.23 seconds
