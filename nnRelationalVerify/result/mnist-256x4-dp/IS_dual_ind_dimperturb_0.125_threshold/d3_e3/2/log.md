## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00058113


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0007763, 0.0011157, 0.0007763, 0.0011157, -0.0003394, 0.0003394)
1: (0.9934265, 0.9942353, 0.9934265, 0.9942353, -0.0008088, 0.0008088)
2: (-0.0085946, -0.0054023, -0.0085946, -0.0054023, -0.0029730, 0.0029730)
3: (0.0036592, 0.0041487, 0.0036592, 0.0041487, -0.0004894, 0.0004894)
4: (0.0026867, 0.0052097, 0.0026867, 0.0052097, -0.0025230, 0.0025230)
5: (0.0052018, 0.0064286, 0.0052018, 0.0064286, -0.0012268, 0.0012268)
6: (-0.0021043, -0.0009831, -0.0021043, -0.0009831, -0.0011213, 0.0011213)
7: (-0.0082500, -0.0075272, -0.0082500, -0.0075272, -0.0007228, 0.0007228)
8: (0.0053495, 0.0095438, 0.0053495, 0.0095438, -0.0041353, 0.0041353)
9: (-0.0036838, -0.0031841, -0.0036838, -0.0031841, -0.0004997, 0.0004997)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.83 + 1.47 = 3.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0006999, upper bound: 0.0006999

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006978, upper bound: 0.0006957
time: 0.60 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006999, upper bound: 0.0006999
time: 0.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.34 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.34
Output dim: 1, lower bound: -0.0006978, upper bound: 0.0006957
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.34
Output dim: 1, lower bound: -0.0006999, upper bound: 0.0006999

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0007595, 0.0011161, 0.0007770, 0.0011156, -0.0003561, 0.0003391
1: 0.9934210, 0.9942709, 0.9934267, 0.9942337, -0.0008127, 0.0008442
2: -0.0086525, -0.0056254, -0.0085920, -0.0054594, -0.0029685, 0.0027410
3: 0.0036382, 0.0041525, 0.0036602, 0.0041485, -0.0005103, 0.0004923
4: 0.0028630, 0.0052555, 0.0027318, 0.0052076, -0.0023447, 0.0025237
5: 0.0051621, 0.0063804, 0.0052036, 0.0064163, -0.0012542, 0.0011768
6: -0.0021245, -0.0009463, -0.0021034, -0.0009847, -0.0011397, 0.0011572
7: -0.0082078, -0.0075102, -0.0082392, -0.0075279, -0.0006799, 0.0007291
8: 0.0056426, 0.0096200, 0.0054246, 0.0095404, -0.0038368, 0.0041347
9: -0.0036815, -0.0031598, -0.0036832, -0.0031852, -0.0004963, 0.0005234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006886, upper bound: 0.0006716
time: 0.64 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006849, upper bound: 0.0006716
time: 0.64 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0007766, 0.0011156, 0.0007763, 0.0011157, -0.0003391, 0.0003394
1: 0.9934265, 0.9942347, 0.9934265, 0.9942353, -0.0008088, 0.0008082
2: -0.0085935, -0.0054283, -0.0085946, -0.0054023, -0.0029719, 0.0029352
3: 0.0036596, 0.0041486, 0.0036592, 0.0041487, -0.0004890, 0.0004894
4: 0.0027072, 0.0052088, 0.0026867, 0.0052097, -0.0025025, 0.0025222
5: 0.0052026, 0.0064230, 0.0052018, 0.0064286, -0.0012261, 0.0012212
6: -0.0021040, -0.0009837, -0.0021043, -0.0009831, -0.0011209, 0.0011206
7: -0.0082451, -0.0075275, -0.0082500, -0.0075272, -0.0007179, 0.0007225
8: 0.0053837, 0.0095424, 0.0053495, 0.0095438, -0.0040983, 0.0041340
9: -0.0036835, -0.0031845, -0.0036838, -0.0031841, -0.0004994, 0.0004993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006957, upper bound: 0.0006978
time: 0.59 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006957, upper bound: 0.0006999
time: 0.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.86 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 1, lower bound: -0.0006886, upper bound: 0.0006716
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 1, lower bound: -0.0006849, upper bound: 0.0006716
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 1, lower bound: -0.0006957, upper bound: 0.0006978
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 1, lower bound: -0.0006957, upper bound: 0.0006999

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007595, 0.0011161, 0.0007772, 0.0011156, -0.0003561, 0.0003389
1: 0.9934210, 0.9942708, 0.9934267, 0.9942334, -0.0008125, 0.0008441
2: -0.0086525, -0.0056338, -0.0085915, -0.0055121, -0.0029135, 0.0027316
3: 0.0036383, 0.0041525, 0.0036604, 0.0041485, -0.0005102, 0.0004921
4: 0.0028697, 0.0052555, 0.0027734, 0.0052072, -0.0023376, 0.0024821
5: 0.0051621, 0.0063786, 0.0052040, 0.0064049, -0.0012428, 0.0011746
6: -0.0021245, -0.0009463, -0.0021033, -0.0009850, -0.0011394, 0.0011569
7: -0.0082062, -0.0075102, -0.0082293, -0.0075281, -0.0006782, 0.0007191
8: 0.0056537, 0.0096199, 0.0054937, 0.0095398, -0.0038251, 0.0040650
9: -0.0036814, -0.0031598, -0.0036826, -0.0031854, -0.0004960, 0.0005229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006403, upper bound: 0.0006469
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006346, upper bound: 0.0006114
time: 0.51 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007595, 0.0011161, 0.0007689, 0.0011158, -0.0003563, 0.0003472
1: 0.9934210, 0.9942708, 0.9934241, 0.9942509, -0.0008299, 0.0008467
2: -0.0086524, -0.0056409, -0.0086200, -0.0055130, -0.0029168, 0.0027590
3: 0.0036383, 0.0041525, 0.0036500, 0.0041504, -0.0005121, 0.0005025
4: 0.0028752, 0.0052554, 0.0027742, 0.0052298, -0.0023546, 0.0024812
5: 0.0051622, 0.0063771, 0.0051844, 0.0064047, -0.0012426, 0.0011927
6: -0.0021244, -0.0009464, -0.0021132, -0.0009669, -0.0011575, 0.0011668
7: -0.0082049, -0.0075102, -0.0082291, -0.0075197, -0.0006852, 0.0007189
8: 0.0056630, 0.0096198, 0.0054951, 0.0095773, -0.0038548, 0.0040648
9: -0.0036813, -0.0031598, -0.0036826, -0.0031734, -0.0005079, 0.0005228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006370, upper bound: 0.0006469
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006310, upper bound: 0.0006114
time: 0.53 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007766, 0.0011156, 0.0007595, 0.0011161, -0.0003395, 0.0003562
1: 0.9934265, 0.9942347, 0.9934210, 0.9942709, -0.0008444, 0.0008137
2: -0.0085935, -0.0054283, -0.0086525, -0.0056254, -0.0027425, 0.0030022
3: 0.0036596, 0.0041486, 0.0036382, 0.0041525, -0.0004929, 0.0005104
4: 0.0027072, 0.0052088, 0.0028630, 0.0052555, -0.0025483, 0.0023459
5: 0.0052026, 0.0064230, 0.0051621, 0.0063804, -0.0011779, 0.0012609
6: -0.0021040, -0.0009837, -0.0021245, -0.0009463, -0.0011577, 0.0011407
7: -0.0082451, -0.0075275, -0.0082078, -0.0075102, -0.0007349, 0.0006804
8: 0.0053837, 0.0095424, 0.0056426, 0.0096200, -0.0041768, 0.0038389
9: -0.0036835, -0.0031845, -0.0036815, -0.0031598, -0.0005237, 0.0004970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006716, upper bound: 0.0006886
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006716, upper bound: 0.0006849
time: 0.70 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007766, 0.0011156, 0.0007766, 0.0011156, -0.0003391, 0.0003391
1: 0.9934265, 0.9942347, 0.9934265, 0.9942347, -0.0008082, 0.0008082
2: -0.0085935, -0.0054283, -0.0085935, -0.0054283, -0.0029342, 0.0029342
3: 0.0036596, 0.0041486, 0.0036596, 0.0041486, -0.0004890, 0.0004890
4: 0.0027072, 0.0052088, 0.0027072, 0.0052088, -0.0025016, 0.0025016
5: 0.0052026, 0.0064230, 0.0052026, 0.0064230, -0.0012204, 0.0012204
6: -0.0021040, -0.0009837, -0.0021040, -0.0009837, -0.0011202, 0.0011202
7: -0.0082451, -0.0075275, -0.0082451, -0.0075275, -0.0007176, 0.0007176
8: 0.0053837, 0.0095424, 0.0053837, 0.0095424, -0.0040969, 0.0040969
9: -0.0036835, -0.0031845, -0.0036835, -0.0031845, -0.0004990, 0.0004990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006716, upper bound: 0.0006997
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006716, upper bound: 0.0006944
time: 0.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.97 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 1, lower bound: -0.0006403, upper bound: 0.0006469
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 1, lower bound: -0.0006346, upper bound: 0.0006114
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 1, lower bound: -0.0006370, upper bound: 0.0006469
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 1, lower bound: -0.0006310, upper bound: 0.0006114
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 1, lower bound: -0.0006716, upper bound: 0.0006886
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 1, lower bound: -0.0006716, upper bound: 0.0006849
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 1, lower bound: -0.0006716, upper bound: 0.0006997
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 1, lower bound: -0.0006716, upper bound: 0.0006944

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007681, 0.0011159, 0.0007772, 0.0011156, -0.0003475, 0.0003387
1: 0.9934238, 0.9942527, 0.9934267, 0.9942334, -0.0008097, 0.0008259
2: -0.0086228, -0.0056351, -0.0085915, -0.0055121, -0.0028853, 0.0027303
3: 0.0036490, 0.0041505, 0.0036604, 0.0041485, -0.0004995, 0.0004902
4: 0.0028707, 0.0052320, 0.0027734, 0.0052072, -0.0023365, 0.0024586
5: 0.0051824, 0.0063783, 0.0052040, 0.0064049, -0.0012225, 0.0011743
6: -0.0021142, -0.0009651, -0.0021033, -0.0009850, -0.0011291, 0.0011381
7: -0.0082060, -0.0075189, -0.0082293, -0.0075281, -0.0006779, 0.0007104
8: 0.0056555, 0.0095810, 0.0054937, 0.0095398, -0.0038233, 0.0040264
9: -0.0036814, -0.0031722, -0.0036826, -0.0031854, -0.0004960, 0.0005104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006256, upper bound: 0.0006354
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006351, upper bound: 0.0006429
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007964, 0.0011151, 0.0007855, 0.0011154, -0.0003190, 0.0003297
1: 0.9934331, 0.9941927, 0.9934295, 0.9942160, -0.0007828, 0.0007632
2: -0.0085250, -0.0054997, -0.0085628, -0.0055138, -0.0028003, 0.0028372
3: 0.0036845, 0.0041441, 0.0036708, 0.0041466, -0.0004621, 0.0004733
4: 0.0027637, 0.0051547, 0.0027748, 0.0051846, -0.0024209, 0.0023799
5: 0.0052496, 0.0064076, 0.0052237, 0.0064045, -0.0011549, 0.0011839
6: -0.0020802, -0.0010272, -0.0020933, -0.0010032, -0.0010770, 0.0010661
7: -0.0082316, -0.0075476, -0.0082289, -0.0075365, -0.0006951, 0.0006813
8: 0.0054776, 0.0094524, 0.0054960, 0.0095021, -0.0039639, 0.0038995
9: -0.0036828, -0.0032132, -0.0036826, -0.0031973, -0.0004854, 0.0004694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006160, upper bound: 0.0006077
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006295, upper bound: 0.0006083
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007681, 0.0011159, 0.0007689, 0.0011158, -0.0003477, 0.0003470
1: 0.9934238, 0.9942525, 0.9934241, 0.9942509, -0.0008271, 0.0008284
2: -0.0086228, -0.0056422, -0.0086200, -0.0055130, -0.0028887, 0.0027576
3: 0.0036491, 0.0041505, 0.0036500, 0.0041504, -0.0005013, 0.0005005
4: 0.0028763, 0.0052320, 0.0027742, 0.0052298, -0.0023535, 0.0024578
5: 0.0051825, 0.0063768, 0.0051844, 0.0064047, -0.0012222, 0.0011924
6: -0.0021141, -0.0009652, -0.0021132, -0.0009669, -0.0011472, 0.0011480
7: -0.0082046, -0.0075189, -0.0082291, -0.0075197, -0.0006849, 0.0007102
8: 0.0056648, 0.0095809, 0.0054951, 0.0095773, -0.0038531, 0.0040262
9: -0.0036813, -0.0031722, -0.0036826, -0.0031734, -0.0005079, 0.0005104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006229, upper bound: 0.0006355
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006322, upper bound: 0.0006428
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007965, 0.0011151, 0.0007772, 0.0011156, -0.0003192, 0.0003380
1: 0.9934331, 0.9941927, 0.9934267, 0.9942335, -0.0008004, 0.0007660
2: -0.0085249, -0.0055051, -0.0085914, -0.0055147, -0.0028035, 0.0028661
3: 0.0036845, 0.0041441, 0.0036604, 0.0041485, -0.0004640, 0.0004837
4: 0.0027679, 0.0051546, 0.0027755, 0.0052072, -0.0024393, 0.0023791
5: 0.0052496, 0.0064064, 0.0052040, 0.0064044, -0.0011547, 0.0012024
6: -0.0020802, -0.0010273, -0.0021033, -0.0009851, -0.0010951, 0.0010760
7: -0.0082306, -0.0075476, -0.0082288, -0.0075281, -0.0007025, 0.0006811
8: 0.0054846, 0.0094523, 0.0054972, 0.0095397, -0.0039959, 0.0038993
9: -0.0036827, -0.0032132, -0.0036826, -0.0031854, -0.0004973, 0.0004694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006146, upper bound: 0.0006077
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006262, upper bound: 0.0006083
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007767, 0.0011156, 0.0007595, 0.0011161, -0.0003394, 0.0003561
1: 0.9934266, 0.9942344, 0.9934210, 0.9942708, -0.0008442, 0.0008134
2: -0.0085930, -0.0054842, -0.0086525, -0.0056338, -0.0027331, 0.0029453
3: 0.0036598, 0.0041486, 0.0036383, 0.0041525, -0.0004927, 0.0005103
4: 0.0027514, 0.0052084, 0.0028697, 0.0052555, -0.0025041, 0.0023388
5: 0.0052029, 0.0064109, 0.0051621, 0.0063786, -0.0011757, 0.0012488
6: -0.0021038, -0.0009841, -0.0021245, -0.0009463, -0.0011575, 0.0011404
7: -0.0082345, -0.0075276, -0.0082062, -0.0075102, -0.0007244, 0.0006786
8: 0.0054571, 0.0095418, 0.0056537, 0.0096199, -0.0041033, 0.0038271
9: -0.0036829, -0.0031847, -0.0036814, -0.0031598, -0.0005231, 0.0004967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006469, upper bound: 0.0006403
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006114, upper bound: 0.0006346
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007685, 0.0011159, 0.0007595, 0.0011161, -0.0003476, 0.0003563
1: 0.9934239, 0.9942518, 0.9934210, 0.9942708, -0.0008469, 0.0008309
2: -0.0086215, -0.0054773, -0.0086524, -0.0056409, -0.0027604, 0.0029553
3: 0.0036495, 0.0041505, 0.0036383, 0.0041525, -0.0005030, 0.0005122
4: 0.0027460, 0.0052310, 0.0028752, 0.0052554, -0.0025094, 0.0023557
5: 0.0051834, 0.0064124, 0.0051622, 0.0063771, -0.0011937, 0.0012503
6: -0.0021137, -0.0009660, -0.0021244, -0.0009464, -0.0011673, 0.0011584
7: -0.0082358, -0.0075193, -0.0082049, -0.0075102, -0.0007256, 0.0006856
8: 0.0054481, 0.0095792, 0.0056630, 0.0096198, -0.0041122, 0.0038568
9: -0.0036830, -0.0031728, -0.0036813, -0.0031598, -0.0005232, 0.0005086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006469, upper bound: 0.0006370
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006114, upper bound: 0.0006310
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007767, 0.0011156, 0.0007766, 0.0011156, -0.0003389, 0.0003390
1: 0.9934266, 0.9942344, 0.9934265, 0.9942346, -0.0008080, 0.0008079
2: -0.0085930, -0.0054842, -0.0085934, -0.0054380, -0.0029240, 0.0028783
3: 0.0036598, 0.0041486, 0.0036597, 0.0041486, -0.0004888, 0.0004889
4: 0.0027514, 0.0052084, 0.0027149, 0.0052088, -0.0024574, 0.0024935
5: 0.0052029, 0.0064109, 0.0052026, 0.0064209, -0.0012180, 0.0012083
6: -0.0021038, -0.0009841, -0.0021039, -0.0009838, -0.0011200, 0.0011199
7: -0.0082345, -0.0075276, -0.0082433, -0.0075275, -0.0007070, 0.0007156
8: 0.0054571, 0.0095418, 0.0053964, 0.0095423, -0.0040235, 0.0040836
9: -0.0036829, -0.0031847, -0.0036834, -0.0031845, -0.0004984, 0.0004987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006835, upper bound: 0.0006922
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006887, upper bound: 0.0006942
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007685, 0.0011159, 0.0007766, 0.0011156, -0.0003472, 0.0003392
1: 0.9934239, 0.9942518, 0.9934266, 0.9942346, -0.0008106, 0.0008252
2: -0.0086215, -0.0054773, -0.0085933, -0.0054414, -0.0029518, 0.0028859
3: 0.0036495, 0.0041505, 0.0036597, 0.0041486, -0.0004991, 0.0004907
4: 0.0027460, 0.0052310, 0.0027176, 0.0052087, -0.0024627, 0.0025134
5: 0.0051834, 0.0064124, 0.0052027, 0.0064202, -0.0012368, 0.0012097
6: -0.0021137, -0.0009660, -0.0021039, -0.0009839, -0.0011298, 0.0011379
7: -0.0082358, -0.0075193, -0.0082426, -0.0075275, -0.0007083, 0.0007234
8: 0.0054481, 0.0095792, 0.0054009, 0.0095421, -0.0040321, 0.0041173
9: -0.0036830, -0.0031728, -0.0036834, -0.0031846, -0.0004984, 0.0005106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006835, upper bound: 0.0006887
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006887, upper bound: 0.0006887
time: 0.68 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.07 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 1, lower bound: -0.0006256, upper bound: 0.0006354
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 1, lower bound: -0.0006351, upper bound: 0.0006429
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 1, lower bound: -0.0006160, upper bound: 0.0006077
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 1, lower bound: -0.0006295, upper bound: 0.0006083
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 1, lower bound: -0.0006229, upper bound: 0.0006355
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 1, lower bound: -0.0006322, upper bound: 0.0006428
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 1, lower bound: -0.0006146, upper bound: 0.0006077
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 1, lower bound: -0.0006262, upper bound: 0.0006083
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 1, lower bound: -0.0006469, upper bound: 0.0006403
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 1, lower bound: -0.0006114, upper bound: 0.0006346
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 1, lower bound: -0.0006469, upper bound: 0.0006370
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 1, lower bound: -0.0006114, upper bound: 0.0006310
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 1, lower bound: -0.0006835, upper bound: 0.0006922
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 1, lower bound: -0.0006887, upper bound: 0.0006942
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 1, lower bound: -0.0006835, upper bound: 0.0006887
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 1, lower bound: -0.0006887, upper bound: 0.0006887

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007728, 0.0011157, 0.0007975, 0.0011151, -0.0003423, 0.0003182
1: 0.9934253, 0.9942426, 0.9934335, 0.9941903, -0.0007650, 0.0008091
2: -0.0086065, -0.0056393, -0.0085212, -0.0055417, -0.0028393, 0.0026544
3: 0.0036550, 0.0041495, 0.0036858, 0.0041438, -0.0004889, 0.0004636
4: 0.0028740, 0.0052191, 0.0027968, 0.0051517, -0.0022777, 0.0024223
5: 0.0051937, 0.0063774, 0.0052522, 0.0063985, -0.0012049, 0.0011252
6: -0.0021085, -0.0009755, -0.0020789, -0.0010296, -0.0010789, 0.0011034
7: -0.0082052, -0.0075237, -0.0082237, -0.0075487, -0.0006565, 0.0007000
8: 0.0056609, 0.0095595, 0.0055327, 0.0094474, -0.0037256, 0.0039660
9: -0.0036813, -0.0031790, -0.0036823, -0.0032148, -0.0004665, 0.0005033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006256, upper bound: 0.0006354
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006256, upper bound: 0.0006354
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007710, 0.0011158, 0.0007902, 0.0011153, -0.0003443, 0.0003255
1: 0.9934247, 0.9942465, 0.9934311, 0.9942057, -0.0007810, 0.0008155
2: -0.0086127, -0.0056373, -0.0085463, -0.0055219, -0.0028655, 0.0026827
3: 0.0036527, 0.0041499, 0.0036767, 0.0041455, -0.0004928, 0.0004731
4: 0.0028724, 0.0052240, 0.0027812, 0.0051716, -0.0022992, 0.0024428
5: 0.0051894, 0.0063779, 0.0052350, 0.0064028, -0.0012134, 0.0011429
6: -0.0021106, -0.0009716, -0.0020876, -0.0010137, -0.0010969, 0.0011160
7: -0.0082056, -0.0075218, -0.0082274, -0.0075413, -0.0006643, 0.0007056
8: 0.0056583, 0.0095677, 0.0055067, 0.0094804, -0.0037614, 0.0039998
9: -0.0036814, -0.0031764, -0.0036825, -0.0032043, -0.0004771, 0.0005061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006351, upper bound: 0.0006429
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006351, upper bound: 0.0006429
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0008012, 0.0011150, 0.0008058, 0.0011149, -0.0003137, 0.0003092
1: 0.9934346, 0.9941825, 0.9934362, 0.9941728, -0.0007382, 0.0007464
2: -0.0085084, -0.0055038, -0.0084926, -0.0055433, -0.0027545, 0.0027620
3: 0.0036905, 0.0041430, 0.0036962, 0.0041419, -0.0004515, 0.0004468
4: 0.0027669, 0.0051416, 0.0027981, 0.0051291, -0.0023622, 0.0023435
5: 0.0052610, 0.0064067, 0.0052718, 0.0063982, -0.0011372, 0.0011349
6: -0.0020744, -0.0010315, -0.0020690, -0.0010452, -0.0010292, 0.0010374
7: -0.0082308, -0.0075525, -0.0082234, -0.0075571, -0.0006737, 0.0006709
8: 0.0054829, 0.0094306, 0.0055348, 0.0094099, -0.0038663, 0.0038390
9: -0.0036827, -0.0032202, -0.0036823, -0.0032268, -0.0004560, 0.0004622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006160, upper bound: 0.0006077
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006160, upper bound: 0.0006077
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007993, 0.0011151, 0.0007987, 0.0011151, -0.0003157, 0.0003164
1: 0.9934340, 0.9941865, 0.9934338, 0.9941878, -0.0007538, 0.0007527
2: -0.0085149, -0.0055018, -0.0085171, -0.0055236, -0.0027807, 0.0027896
3: 0.0036881, 0.0041434, 0.0036873, 0.0041435, -0.0004554, 0.0004561
4: 0.0027653, 0.0051467, 0.0027826, 0.0051485, -0.0023832, 0.0023641
5: 0.0052565, 0.0064071, 0.0052550, 0.0064024, -0.0011459, 0.0011522
6: -0.0020767, -0.0010308, -0.0020775, -0.0010322, -0.0010445, 0.0010467
7: -0.0082312, -0.0075505, -0.0082271, -0.0075499, -0.0006813, 0.0006765
8: 0.0054803, 0.0094392, 0.0055089, 0.0094421, -0.0039014, 0.0038733
9: -0.0036827, -0.0032174, -0.0036825, -0.0032165, -0.0004663, 0.0004651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006295, upper bound: 0.0006083
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006295, upper bound: 0.0006083
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007728, 0.0011157, 0.0007890, 0.0011153, -0.0003425, 0.0003268
1: 0.9934254, 0.9942425, 0.9934306, 0.9942084, -0.0007830, 0.0008119
2: -0.0086064, -0.0056463, -0.0085506, -0.0055360, -0.0028451, 0.0026789
3: 0.0036550, 0.0041495, 0.0036752, 0.0041458, -0.0004908, 0.0004743
4: 0.0028796, 0.0052190, 0.0027924, 0.0051750, -0.0022954, 0.0024266
5: 0.0051937, 0.0063759, 0.0052320, 0.0063997, -0.0012060, 0.0011439
6: -0.0021084, -0.0009756, -0.0020891, -0.0010109, -0.0010975, 0.0011135
7: -0.0082039, -0.0075237, -0.0082247, -0.0075401, -0.0006638, 0.0007010
8: 0.0056702, 0.0095594, 0.0055253, 0.0094861, -0.0037550, 0.0039738
9: -0.0036813, -0.0031791, -0.0036824, -0.0032025, -0.0004788, 0.0005033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005979, upper bound: 0.0006043
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006228, upper bound: 0.0006355
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007711, 0.0011158, 0.0007815, 0.0011155, -0.0003445, 0.0003343
1: 0.9934247, 0.9942464, 0.9934281, 0.9942243, -0.0007995, 0.0008183
2: -0.0086126, -0.0056443, -0.0085765, -0.0055230, -0.0028687, 0.0027110
3: 0.0036527, 0.0041499, 0.0036658, 0.0041475, -0.0004948, 0.0004841
4: 0.0028780, 0.0052240, 0.0027820, 0.0051954, -0.0023174, 0.0024419
5: 0.0051895, 0.0063763, 0.0052142, 0.0064026, -0.0012131, 0.0011621
6: -0.0021106, -0.0009716, -0.0020981, -0.0009945, -0.0011161, 0.0011265
7: -0.0082042, -0.0075218, -0.0082272, -0.0075324, -0.0006718, 0.0007054
8: 0.0056676, 0.0095676, 0.0055081, 0.0095201, -0.0037924, 0.0039999
9: -0.0036813, -0.0031765, -0.0036825, -0.0031916, -0.0004897, 0.0005060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006032, upper bound: 0.0006086
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006321, upper bound: 0.0006428
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0008012, 0.0011150, 0.0007973, 0.0011151, -0.0003139, 0.0003177
1: 0.9934346, 0.9941824, 0.9934334, 0.9941908, -0.0007562, 0.0007491
2: -0.0085083, -0.0055092, -0.0085220, -0.0055375, -0.0027601, 0.0027874
3: 0.0036905, 0.0041430, 0.0036855, 0.0041439, -0.0004534, 0.0004574
4: 0.0027712, 0.0051415, 0.0027935, 0.0051524, -0.0023812, 0.0023480
5: 0.0052610, 0.0064055, 0.0052516, 0.0063994, -0.0011384, 0.0011539
6: -0.0020744, -0.0010334, -0.0020792, -0.0010291, -0.0010453, 0.0010458
7: -0.0082298, -0.0075525, -0.0082244, -0.0075485, -0.0006813, 0.0006720
8: 0.0054900, 0.0094305, 0.0055272, 0.0094485, -0.0038979, 0.0038468
9: -0.0036827, -0.0032202, -0.0036824, -0.0032144, -0.0004682, 0.0004622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006146, upper bound: 0.0006077
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006146, upper bound: 0.0006077
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007994, 0.0011151, 0.0007897, 0.0011153, -0.0003159, 0.0003253
1: 0.9934340, 0.9941864, 0.9934309, 0.9942069, -0.0007729, 0.0007555
2: -0.0085148, -0.0055072, -0.0085482, -0.0055245, -0.0027838, 0.0028195
3: 0.0036881, 0.0041434, 0.0036761, 0.0041456, -0.0004575, 0.0004673
4: 0.0027696, 0.0051467, 0.0027833, 0.0051730, -0.0024035, 0.0023634
5: 0.0052566, 0.0064060, 0.0052337, 0.0064022, -0.0011457, 0.0011723
6: -0.0020767, -0.0010327, -0.0020882, -0.0010125, -0.0010642, 0.0010556
7: -0.0082302, -0.0075506, -0.0082269, -0.0075408, -0.0006894, 0.0006763
8: 0.0054874, 0.0094390, 0.0055102, 0.0094829, -0.0039363, 0.0038735
9: -0.0036827, -0.0032175, -0.0036825, -0.0032035, -0.0004792, 0.0004651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005668, upper bound: 0.0005491
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005668, upper bound: 0.0006083
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007767, 0.0011156, 0.0007681, 0.0011159, -0.0003391, 0.0003475
1: 0.9934266, 0.9942344, 0.9934238, 0.9942527, -0.0008261, 0.0008106
2: -0.0085930, -0.0054842, -0.0086228, -0.0056351, -0.0027318, 0.0029172
3: 0.0036598, 0.0041486, 0.0036490, 0.0041505, -0.0004907, 0.0004996
4: 0.0027514, 0.0052084, 0.0028707, 0.0052320, -0.0024807, 0.0023377
5: 0.0052029, 0.0064109, 0.0051824, 0.0063783, -0.0011754, 0.0012285
6: -0.0021038, -0.0009841, -0.0021142, -0.0009651, -0.0011387, 0.0011301
7: -0.0082345, -0.0075276, -0.0082060, -0.0075189, -0.0007157, 0.0006783
8: 0.0054571, 0.0095418, 0.0056555, 0.0095810, -0.0040647, 0.0038253
9: -0.0036829, -0.0031847, -0.0036814, -0.0031722, -0.0005107, 0.0004967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006354, upper bound: 0.0006256
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006429, upper bound: 0.0006351
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007850, 0.0011154, 0.0007964, 0.0011151, -0.0003301, 0.0003190
1: 0.9934293, 0.9942168, 0.9934331, 0.9941927, -0.0007634, 0.0007837
2: -0.0085643, -0.0054860, -0.0085250, -0.0054997, -0.0028387, 0.0028321
3: 0.0036702, 0.0041467, 0.0036845, 0.0041441, -0.0004739, 0.0004622
4: 0.0027528, 0.0051858, 0.0027637, 0.0051547, -0.0024019, 0.0024221
5: 0.0052226, 0.0064106, 0.0052496, 0.0064076, -0.0011850, 0.0011610
6: -0.0020938, -0.0010022, -0.0020802, -0.0010272, -0.0010666, 0.0010779
7: -0.0082342, -0.0075360, -0.0082316, -0.0075476, -0.0006866, 0.0006955
8: 0.0054595, 0.0095041, 0.0054776, 0.0094524, -0.0039377, 0.0039659
9: -0.0036829, -0.0031967, -0.0036828, -0.0032132, -0.0004697, 0.0004860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006078, upper bound: 0.0006160
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006295
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007685, 0.0011159, 0.0007681, 0.0011159, -0.0003474, 0.0003477
1: 0.9934239, 0.9942518, 0.9934238, 0.9942525, -0.0008286, 0.0008281
2: -0.0086215, -0.0054773, -0.0086228, -0.0056422, -0.0027591, 0.0029272
3: 0.0036495, 0.0041505, 0.0036491, 0.0041505, -0.0005010, 0.0005014
4: 0.0027460, 0.0052310, 0.0028763, 0.0052320, -0.0024860, 0.0023547
5: 0.0051834, 0.0064124, 0.0051825, 0.0063768, -0.0011934, 0.0012299
6: -0.0021137, -0.0009660, -0.0021141, -0.0009652, -0.0011485, 0.0011481
7: -0.0082358, -0.0075193, -0.0082046, -0.0075189, -0.0007169, 0.0006854
8: 0.0054481, 0.0095792, 0.0056648, 0.0095809, -0.0040736, 0.0038550
9: -0.0036830, -0.0031728, -0.0036813, -0.0031722, -0.0005108, 0.0005086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006355, upper bound: 0.0006229
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006428, upper bound: 0.0006323
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007767, 0.0011156, 0.0007965, 0.0011151, -0.0003384, 0.0003192
1: 0.9934266, 0.9942343, 0.9934331, 0.9941927, -0.0007661, 0.0008011
2: -0.0085929, -0.0054790, -0.0085249, -0.0055051, -0.0028675, 0.0028421
3: 0.0036599, 0.0041486, 0.0036845, 0.0041441, -0.0004842, 0.0004641
4: 0.0027473, 0.0052084, 0.0027679, 0.0051546, -0.0024073, 0.0024405
5: 0.0052030, 0.0064121, 0.0052496, 0.0064064, -0.0012034, 0.0011624
6: -0.0021038, -0.0009841, -0.0020802, -0.0010273, -0.0010765, 0.0010961
7: -0.0082355, -0.0075277, -0.0082306, -0.0075476, -0.0006879, 0.0007029
8: 0.0054504, 0.0095417, 0.0054846, 0.0094523, -0.0039467, 0.0039979
9: -0.0036830, -0.0031847, -0.0036827, -0.0032132, -0.0004697, 0.0004980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006077, upper bound: 0.0006146
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006262
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007816, 0.0011155, 0.0007970, 0.0011151, -0.0003335, 0.0003186
1: 0.9934282, 0.9942241, 0.9934332, 0.9941916, -0.0007634, 0.0007908
2: -0.0085761, -0.0054882, -0.0085231, -0.0054652, -0.0028765, 0.0028030
3: 0.0036660, 0.0041474, 0.0036851, 0.0041439, -0.0004780, 0.0004623
4: 0.0027545, 0.0051951, 0.0027364, 0.0051532, -0.0023987, 0.0024587
5: 0.0052145, 0.0064101, 0.0052509, 0.0064150, -0.0012005, 0.0011592
6: -0.0020979, -0.0009948, -0.0020795, -0.0010181, -0.0010798, 0.0010847
7: -0.0082338, -0.0075326, -0.0082381, -0.0075481, -0.0006856, 0.0007055
8: 0.0054624, 0.0095195, 0.0054322, 0.0094499, -0.0039257, 0.0040245
9: -0.0036829, -0.0031918, -0.0036831, -0.0032140, -0.0004689, 0.0004913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006664
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006472
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007796, 0.0011156, 0.0007897, 0.0011153, -0.0003357, 0.0003259
1: 0.9934275, 0.9942282, 0.9934309, 0.9942070, -0.0007794, 0.0007974
2: -0.0085831, -0.0054861, -0.0085482, -0.0054474, -0.0029059, 0.0028293
3: 0.0036634, 0.0041479, 0.0036761, 0.0041456, -0.0004822, 0.0004719
4: 0.0027529, 0.0052006, 0.0027223, 0.0051731, -0.0024201, 0.0024783
5: 0.0052097, 0.0064105, 0.0052336, 0.0064189, -0.0012092, 0.0011769
6: -0.0021004, -0.0009903, -0.0020883, -0.0010119, -0.0010884, 0.0010979
7: -0.0082342, -0.0075305, -0.0082415, -0.0075408, -0.0006934, 0.0007109
8: 0.0054597, 0.0095288, 0.0054088, 0.0094830, -0.0039612, 0.0040582
9: -0.0036829, -0.0031888, -0.0036833, -0.0032035, -0.0004794, 0.0004945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006672
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006477
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007734, 0.0011157, 0.0007970, 0.0011151, -0.0003417, 0.0003187
1: 0.9934255, 0.9942415, 0.9934332, 0.9941913, -0.0007658, 0.0008082
2: -0.0086045, -0.0054814, -0.0085229, -0.0054691, -0.0029031, 0.0028107
3: 0.0036556, 0.0041493, 0.0036852, 0.0041439, -0.0004883, 0.0004641
4: 0.0027492, 0.0052176, 0.0027395, 0.0051531, -0.0024039, 0.0024781
5: 0.0051950, 0.0064115, 0.0052510, 0.0064142, -0.0012192, 0.0011606
6: -0.0021078, -0.0009768, -0.0020795, -0.0010195, -0.0010883, 0.0011027
7: -0.0082351, -0.0075242, -0.0082374, -0.0075482, -0.0006869, 0.0007131
8: 0.0054535, 0.0095569, 0.0054374, 0.0094497, -0.0039342, 0.0040568
9: -0.0036830, -0.0031799, -0.0036831, -0.0032141, -0.0004689, 0.0005032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006656
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006432
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007710, 0.0011158, 0.0007897, 0.0011153, -0.0003443, 0.0003261
1: 0.9934248, 0.9942464, 0.9934309, 0.9942068, -0.0007821, 0.0008155
2: -0.0086126, -0.0054793, -0.0085481, -0.0054508, -0.0029339, 0.0028371
3: 0.0036527, 0.0041499, 0.0036761, 0.0041456, -0.0004929, 0.0004738
4: 0.0027475, 0.0052239, 0.0027251, 0.0051730, -0.0024254, 0.0024989
5: 0.0051895, 0.0064120, 0.0052337, 0.0064181, -0.0012286, 0.0011783
6: -0.0021106, -0.0009716, -0.0020882, -0.0010126, -0.0010980, 0.0011166
7: -0.0082355, -0.0075219, -0.0082408, -0.0075408, -0.0006946, 0.0007190
8: 0.0054507, 0.0095675, 0.0054133, 0.0094828, -0.0039699, 0.0040931
9: -0.0036830, -0.0031765, -0.0036833, -0.0032035, -0.0004795, 0.0005068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006657
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006432
time: 0.60 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.05 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006256, upper bound: 0.0006354
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006256, upper bound: 0.0006354
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006351, upper bound: 0.0006429
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006351, upper bound: 0.0006429
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006160, upper bound: 0.0006077
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006160, upper bound: 0.0006077
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006295, upper bound: 0.0006083
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006295, upper bound: 0.0006083
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0005979, upper bound: 0.0006043
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006228, upper bound: 0.0006355
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006032, upper bound: 0.0006086
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006321, upper bound: 0.0006428
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006146, upper bound: 0.0006077
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006146, upper bound: 0.0006077
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0005668, upper bound: 0.0005491
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0005668, upper bound: 0.0006083
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006354, upper bound: 0.0006256
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006429, upper bound: 0.0006351
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006078, upper bound: 0.0006160
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006295
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006355, upper bound: 0.0006229
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006428, upper bound: 0.0006323
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006077, upper bound: 0.0006146
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006262
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006664
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006472
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006672
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006477
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006656
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006432
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006657
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006432

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007729, 0.0011157, 0.0007975, 0.0011151, -0.0003422, 0.0003182
1: 0.9934254, 0.9942424, 0.9934335, 0.9941903, -0.0007650, 0.0008090
2: -0.0086062, -0.0056797, -0.0085212, -0.0055417, -0.0028390, 0.0026132
3: 0.0036550, 0.0041494, 0.0036858, 0.0041438, -0.0004888, 0.0004636
4: 0.0029060, 0.0052189, 0.0027968, 0.0051517, -0.0022457, 0.0024221
5: 0.0051939, 0.0063687, 0.0052522, 0.0063985, -0.0012046, 0.0011165
6: -0.0021084, -0.0009757, -0.0020789, -0.0010296, -0.0010788, 0.0011032
7: -0.0081976, -0.0075238, -0.0082237, -0.0075487, -0.0006489, 0.0006999
8: 0.0057140, 0.0095591, 0.0055327, 0.0094474, -0.0036723, 0.0039657
9: -0.0036809, -0.0031792, -0.0036823, -0.0032148, -0.0004661, 0.0005032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006256, upper bound: 0.0006354
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006256, upper bound: 0.0006354
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007669, 0.0011159, 0.0007975, 0.0011151, -0.0003482, 0.0003184
1: 0.9934233, 0.9942552, 0.9934335, 0.9941903, -0.0007670, 0.0008217
2: -0.0086270, -0.0056870, -0.0085212, -0.0055417, -0.0028640, 0.0026085
3: 0.0036475, 0.0041508, 0.0036858, 0.0041438, -0.0004963, 0.0004650
4: 0.0029117, 0.0052353, 0.0027968, 0.0051517, -0.0022400, 0.0024385
5: 0.0051796, 0.0063671, 0.0052522, 0.0063985, -0.0012189, 0.0011149
6: -0.0021156, -0.0009625, -0.0020789, -0.0010296, -0.0010860, 0.0011164
7: -0.0081962, -0.0075177, -0.0082237, -0.0075487, -0.0006475, 0.0007060
8: 0.0057236, 0.0095865, 0.0055327, 0.0094474, -0.0036635, 0.0039937
9: -0.0036808, -0.0031704, -0.0036823, -0.0032148, -0.0004661, 0.0005119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006256, upper bound: 0.0006354
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006256, upper bound: 0.0006354
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007711, 0.0011158, 0.0007902, 0.0011153, -0.0003442, 0.0003255
1: 0.9934248, 0.9942463, 0.9934311, 0.9942057, -0.0007809, 0.0008152
2: -0.0086124, -0.0056777, -0.0085463, -0.0055219, -0.0028652, 0.0026416
3: 0.0036528, 0.0041499, 0.0036767, 0.0041455, -0.0004927, 0.0004731
4: 0.0029044, 0.0052238, 0.0027812, 0.0051716, -0.0022672, 0.0024426
5: 0.0051896, 0.0063691, 0.0052350, 0.0064028, -0.0012132, 0.0011342
6: -0.0021105, -0.0009717, -0.0020876, -0.0010137, -0.0010969, 0.0011158
7: -0.0081979, -0.0075219, -0.0082274, -0.0075413, -0.0006566, 0.0007055
8: 0.0057114, 0.0095673, 0.0055067, 0.0094804, -0.0037082, 0.0039995
9: -0.0036809, -0.0031766, -0.0036825, -0.0032043, -0.0004767, 0.0005060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006351, upper bound: 0.0006429
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006351, upper bound: 0.0006429
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007651, 0.0011159, 0.0007902, 0.0011153, -0.0003502, 0.0003257
1: 0.9934227, 0.9942590, 0.9934311, 0.9942057, -0.0007830, 0.0008279
2: -0.0086331, -0.0056850, -0.0085463, -0.0055219, -0.0028894, 0.0026368
3: 0.0036453, 0.0041512, 0.0036767, 0.0041455, -0.0005002, 0.0004745
4: 0.0029101, 0.0052401, 0.0027812, 0.0051716, -0.0022615, 0.0024589
5: 0.0051755, 0.0063675, 0.0052350, 0.0064028, -0.0012273, 0.0011326
6: -0.0021177, -0.0009586, -0.0020876, -0.0010137, -0.0011040, 0.0011289
7: -0.0081966, -0.0075159, -0.0082274, -0.0075413, -0.0006552, 0.0007115
8: 0.0057209, 0.0095944, 0.0055067, 0.0094804, -0.0036996, 0.0040276
9: -0.0036809, -0.0031679, -0.0036825, -0.0032043, -0.0004766, 0.0005146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006351, upper bound: 0.0006429
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006351, upper bound: 0.0006429
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0008013, 0.0011150, 0.0008058, 0.0011149, -0.0003136, 0.0003092
1: 0.9934347, 0.9941822, 0.9934362, 0.9941728, -0.0007381, 0.0007460
2: -0.0085080, -0.0055472, -0.0084926, -0.0055433, -0.0027541, 0.0027187
3: 0.0036906, 0.0041429, 0.0036962, 0.0041419, -0.0004513, 0.0004468
4: 0.0028013, 0.0051413, 0.0027981, 0.0051291, -0.0023278, 0.0023432
5: 0.0052612, 0.0063973, 0.0052718, 0.0063982, -0.0011370, 0.0011255
6: -0.0020743, -0.0010379, -0.0020690, -0.0010452, -0.0010291, 0.0010310
7: -0.0082226, -0.0075526, -0.0082234, -0.0075571, -0.0006655, 0.0006708
8: 0.0055400, 0.0094302, 0.0055348, 0.0094099, -0.0038093, 0.0038385
9: -0.0036823, -0.0032203, -0.0036823, -0.0032268, -0.0004555, 0.0004620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005945, upper bound: 0.0005935
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005945, upper bound: 0.0006078
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007947, 0.0011152, 0.0008058, 0.0011149, -0.0003202, 0.0003094
1: 0.9934325, 0.9941963, 0.9934362, 0.9941728, -0.0007403, 0.0007601
2: -0.0085309, -0.0055494, -0.0084926, -0.0055433, -0.0027800, 0.0027162
3: 0.0036823, 0.0041445, 0.0036962, 0.0041419, -0.0004596, 0.0004483
4: 0.0028030, 0.0051594, 0.0027981, 0.0051291, -0.0023261, 0.0023612
5: 0.0052455, 0.0063968, 0.0052718, 0.0063982, -0.0011526, 0.0011250
6: -0.0020822, -0.0010235, -0.0020690, -0.0010452, -0.0010370, 0.0010455
7: -0.0082222, -0.0075459, -0.0082234, -0.0075571, -0.0006651, 0.0006775
8: 0.0055428, 0.0094602, 0.0055348, 0.0094099, -0.0038064, 0.0038695
9: -0.0036823, -0.0032107, -0.0036823, -0.0032268, -0.0004555, 0.0004716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005945, upper bound: 0.0005935
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005945, upper bound: 0.0005935
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007994, 0.0011151, 0.0007987, 0.0011151, -0.0003156, 0.0003164
1: 0.9934340, 0.9941863, 0.9934338, 0.9941878, -0.0007538, 0.0007525
2: -0.0085146, -0.0055453, -0.0085171, -0.0055236, -0.0027804, 0.0027463
3: 0.0036882, 0.0041434, 0.0036873, 0.0041435, -0.0004553, 0.0004561
4: 0.0027997, 0.0051465, 0.0027826, 0.0051485, -0.0023488, 0.0023639
5: 0.0052567, 0.0063977, 0.0052550, 0.0064024, -0.0011457, 0.0011428
6: -0.0020766, -0.0010338, -0.0020775, -0.0010322, -0.0010444, 0.0010436
7: -0.0082230, -0.0075507, -0.0082271, -0.0075499, -0.0006731, 0.0006764
8: 0.0055374, 0.0094387, 0.0055089, 0.0094421, -0.0038445, 0.0038729
9: -0.0036823, -0.0032176, -0.0036825, -0.0032165, -0.0004658, 0.0004649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006265, upper bound: 0.0006083
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006265, upper bound: 0.0006083
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007930, 0.0011152, 0.0007987, 0.0011151, -0.0003221, 0.0003165
1: 0.9934319, 0.9942000, 0.9934338, 0.9941878, -0.0007559, 0.0007662
2: -0.0085369, -0.0055474, -0.0085171, -0.0055236, -0.0028059, 0.0027438
3: 0.0036801, 0.0041449, 0.0036873, 0.0041435, -0.0004634, 0.0004576
4: 0.0028014, 0.0051641, 0.0027826, 0.0051485, -0.0023471, 0.0023816
5: 0.0052414, 0.0063973, 0.0052550, 0.0064024, -0.0011610, 0.0011423
6: -0.0020843, -0.0010196, -0.0020775, -0.0010322, -0.0010521, 0.0010578
7: -0.0082226, -0.0075441, -0.0082271, -0.0075499, -0.0006727, 0.0006830
8: 0.0055402, 0.0094681, 0.0055089, 0.0094421, -0.0038415, 0.0039031
9: -0.0036823, -0.0032082, -0.0036825, -0.0032165, -0.0004658, 0.0004743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006265, upper bound: 0.0006083
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006265, upper bound: 0.0006083
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007677, 0.0011159, 0.0007891, 0.0011153, -0.0003476, 0.0003267
1: 0.9934236, 0.9942535, 0.9934307, 0.9942081, -0.0007845, 0.0008228
2: -0.0086242, -0.0057134, -0.0085502, -0.0055527, -0.0028492, 0.0026139
3: 0.0036485, 0.0041506, 0.0036753, 0.0041457, -0.0004972, 0.0004753
4: 0.0029325, 0.0052331, 0.0028056, 0.0051746, -0.0022421, 0.0024276
5: 0.0051815, 0.0063614, 0.0052323, 0.0063961, -0.0012146, 0.0011291
6: -0.0021146, -0.0009643, -0.0020889, -0.0010112, -0.0011034, 0.0011247
7: -0.0081912, -0.0075185, -0.0082216, -0.0075402, -0.0006510, 0.0006992
8: 0.0057583, 0.0095828, 0.0055472, 0.0094855, -0.0036673, 0.0039759
9: -0.0036806, -0.0031716, -0.0036822, -0.0032026, -0.0004779, 0.0005106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005979, upper bound: 0.0006043
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005979, upper bound: 0.0006043
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007736, 0.0011157, 0.0007890, 0.0011153, -0.0003418, 0.0003267
1: 0.9934255, 0.9942411, 0.9934306, 0.9942083, -0.0007828, 0.0008105
2: -0.0086039, -0.0056800, -0.0085505, -0.0055383, -0.0028406, 0.0026458
3: 0.0036559, 0.0041493, 0.0036752, 0.0041458, -0.0004899, 0.0004741
4: 0.0029062, 0.0052171, 0.0027942, 0.0051749, -0.0022687, 0.0024230
5: 0.0051954, 0.0063686, 0.0052321, 0.0063992, -0.0012038, 0.0011365
6: -0.0021076, -0.0009771, -0.0020890, -0.0010110, -0.0010966, 0.0011119
7: -0.0081975, -0.0075244, -0.0082243, -0.0075401, -0.0006574, 0.0006988
8: 0.0057145, 0.0095562, 0.0055282, 0.0094859, -0.0037107, 0.0039676
9: -0.0036809, -0.0031801, -0.0036824, -0.0032025, -0.0004784, 0.0005023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006228, upper bound: 0.0006355
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006228, upper bound: 0.0006355
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007658, 0.0011159, 0.0007816, 0.0011155, -0.0003497, 0.0003343
1: 0.9934230, 0.9942575, 0.9934282, 0.9942240, -0.0008010, 0.0008292
2: -0.0086308, -0.0057112, -0.0085761, -0.0055400, -0.0028720, 0.0026463
3: 0.0036461, 0.0041511, 0.0036659, 0.0041475, -0.0005013, 0.0004851
4: 0.0029308, 0.0052383, 0.0027956, 0.0051951, -0.0022642, 0.0024428
5: 0.0051770, 0.0063619, 0.0052145, 0.0063989, -0.0012219, 0.0011474
6: -0.0021169, -0.0009601, -0.0020979, -0.0009948, -0.0011221, 0.0011378
7: -0.0081916, -0.0075165, -0.0082240, -0.0075326, -0.0006590, 0.0007074
8: 0.0057554, 0.0095914, 0.0055305, 0.0095196, -0.0037051, 0.0040018
9: -0.0036806, -0.0031689, -0.0036824, -0.0031918, -0.0004888, 0.0005135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006032, upper bound: 0.0006086
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006032, upper bound: 0.0006086
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007718, 0.0011158, 0.0007815, 0.0011155, -0.0003438, 0.0003342
1: 0.9934250, 0.9942449, 0.9934282, 0.9942241, -0.0007991, 0.0008167
2: -0.0086101, -0.0056780, -0.0085764, -0.0055251, -0.0028642, 0.0026787
3: 0.0036536, 0.0041497, 0.0036658, 0.0041475, -0.0004939, 0.0004839
4: 0.0029046, 0.0052220, 0.0027837, 0.0051953, -0.0022907, 0.0024383
5: 0.0051912, 0.0063690, 0.0052143, 0.0064021, -0.0012109, 0.0011547
6: -0.0021098, -0.0009732, -0.0020980, -0.0009946, -0.0011152, 0.0011248
7: -0.0081979, -0.0075226, -0.0082268, -0.0075325, -0.0006654, 0.0007042
8: 0.0057118, 0.0095643, 0.0055109, 0.0095200, -0.0037485, 0.0039939
9: -0.0036809, -0.0031775, -0.0036825, -0.0031917, -0.0004893, 0.0005050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006321, upper bound: 0.0006428
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006321, upper bound: 0.0006428
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0008013, 0.0011150, 0.0007973, 0.0011151, -0.0003138, 0.0003177
1: 0.9934347, 0.9941822, 0.9934334, 0.9941908, -0.0007561, 0.0007488
2: -0.0085080, -0.0055472, -0.0085220, -0.0055375, -0.0027606, 0.0027497
3: 0.0036906, 0.0041429, 0.0036855, 0.0041439, -0.0004533, 0.0004574
4: 0.0028013, 0.0051413, 0.0027935, 0.0051524, -0.0023511, 0.0023478
5: 0.0052612, 0.0063973, 0.0052516, 0.0063994, -0.0011382, 0.0011457
6: -0.0020743, -0.0010379, -0.0020792, -0.0010291, -0.0010452, 0.0010412
7: -0.0082226, -0.0075526, -0.0082244, -0.0075485, -0.0006741, 0.0006719
8: 0.0055400, 0.0094302, 0.0055272, 0.0094485, -0.0038480, 0.0038467
9: -0.0036823, -0.0032203, -0.0036824, -0.0032144, -0.0004678, 0.0004621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005550
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005550
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007947, 0.0011152, 0.0007973, 0.0011151, -0.0003204, 0.0003179
1: 0.9934325, 0.9941963, 0.9934334, 0.9941908, -0.0007583, 0.0007629
2: -0.0085309, -0.0055494, -0.0085220, -0.0055375, -0.0027824, 0.0027435
3: 0.0036823, 0.0041445, 0.0036855, 0.0041439, -0.0004616, 0.0004589
4: 0.0028030, 0.0051594, 0.0027935, 0.0051524, -0.0023494, 0.0023658
5: 0.0052455, 0.0063968, 0.0052516, 0.0063994, -0.0011539, 0.0011452
6: -0.0020822, -0.0010235, -0.0020792, -0.0010291, -0.0010531, 0.0010557
7: -0.0082222, -0.0075459, -0.0082244, -0.0075485, -0.0006737, 0.0006786
8: 0.0055428, 0.0094602, 0.0055272, 0.0094485, -0.0038444, 0.0038765
9: -0.0036823, -0.0032107, -0.0036824, -0.0032144, -0.0004678, 0.0004716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005550
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005550
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0008106, 0.0011148, 0.0007897, 0.0011153, -0.0003047, 0.0003251
1: 0.9934377, 0.9941624, 0.9934309, 0.9942069, -0.0007692, 0.0007315
2: -0.0084758, -0.0055152, -0.0085482, -0.0055245, -0.0027412, 0.0028116
3: 0.0037023, 0.0041408, 0.0036761, 0.0041456, -0.0004433, 0.0004648
4: 0.0027759, 0.0051158, 0.0027833, 0.0051730, -0.0023971, 0.0023325
5: 0.0052833, 0.0064042, 0.0052337, 0.0064022, -0.0011189, 0.0011706
6: -0.0020631, -0.0010355, -0.0020882, -0.0010125, -0.0010506, 0.0010528
7: -0.0082287, -0.0075620, -0.0082269, -0.0075408, -0.0006879, 0.0006649
8: 0.0054979, 0.0093878, 0.0055102, 0.0094829, -0.0039258, 0.0038212
9: -0.0036826, -0.0032338, -0.0036825, -0.0032035, -0.0004791, 0.0004487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005362, upper bound: 0.0006083
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005362, upper bound: 0.0005362
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007971, 0.0011151, 0.0007728, 0.0011157, -0.0003187, 0.0003423
1: 0.9934333, 0.9941913, 0.9934253, 0.9942426, -0.0008093, 0.0007660
2: -0.0085227, -0.0055119, -0.0086065, -0.0056393, -0.0026558, 0.0028687
3: 0.0036853, 0.0041439, 0.0036550, 0.0041495, -0.0004642, 0.0004890
4: 0.0027733, 0.0051529, 0.0028740, 0.0052191, -0.0024458, 0.0022789
5: 0.0052512, 0.0064049, 0.0051937, 0.0063774, -0.0011262, 0.0012113
6: -0.0020794, -0.0010287, -0.0021085, -0.0009755, -0.0011039, 0.0010798
7: -0.0082293, -0.0075483, -0.0082052, -0.0075237, -0.0007056, 0.0006569
8: 0.0054936, 0.0094493, 0.0056609, 0.0095595, -0.0040054, 0.0037275
9: -0.0036826, -0.0032142, -0.0036813, -0.0031790, -0.0005036, 0.0004671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006354, upper bound: 0.0006256
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006354, upper bound: 0.0006256
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007898, 0.0011153, 0.0007710, 0.0011158, -0.0003260, 0.0003443
1: 0.9934309, 0.9942066, 0.9934247, 0.9942465, -0.0008156, 0.0007819
2: -0.0085478, -0.0054935, -0.0086127, -0.0056373, -0.0026843, 0.0028975
3: 0.0036762, 0.0041456, 0.0036527, 0.0041499, -0.0004737, 0.0004929
4: 0.0027588, 0.0051727, 0.0028724, 0.0052240, -0.0024652, 0.0023003
5: 0.0052339, 0.0064089, 0.0051894, 0.0063779, -0.0011439, 0.0012195
6: -0.0020881, -0.0010127, -0.0021106, -0.0009716, -0.0011165, 0.0010979
7: -0.0082328, -0.0075409, -0.0082056, -0.0075218, -0.0007109, 0.0006647
8: 0.0054694, 0.0094824, 0.0056583, 0.0095677, -0.0040388, 0.0037634
9: -0.0036828, -0.0032036, -0.0036814, -0.0031764, -0.0005064, 0.0004777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006429, upper bound: 0.0006351
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006429, upper bound: 0.0006351
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0008054, 0.0011149, 0.0008012, 0.0011150, -0.0003096, 0.0003137
1: 0.9934360, 0.9941737, 0.9934346, 0.9941825, -0.0007465, 0.0007391
2: -0.0084941, -0.0055135, -0.0085084, -0.0055038, -0.0027634, 0.0027839
3: 0.0036957, 0.0041420, 0.0036905, 0.0041430, -0.0004473, 0.0004516
4: 0.0027746, 0.0051303, 0.0027669, 0.0051416, -0.0023670, 0.0023633
5: 0.0052708, 0.0064046, 0.0052610, 0.0064067, -0.0011359, 0.0011436
6: -0.0020695, -0.0010349, -0.0020744, -0.0010315, -0.0010379, 0.0010396
7: -0.0082290, -0.0075567, -0.0082308, -0.0075525, -0.0006765, 0.0006742
8: 0.0054957, 0.0094118, 0.0054829, 0.0094306, -0.0038784, 0.0038682
9: -0.0036826, -0.0032262, -0.0036827, -0.0032202, -0.0004625, 0.0004566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006078, upper bound: 0.0006160
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006078, upper bound: 0.0006160
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007982, 0.0011151, 0.0007993, 0.0011151, -0.0003168, 0.0003157
1: 0.9934337, 0.9941887, 0.9934340, 0.9941865, -0.0007528, 0.0007547
2: -0.0085186, -0.0054952, -0.0085149, -0.0055018, -0.0027912, 0.0028127
3: 0.0036867, 0.0041436, 0.0036881, 0.0041434, -0.0004567, 0.0004555
4: 0.0027601, 0.0051497, 0.0027653, 0.0051467, -0.0023866, 0.0023843
5: 0.0052539, 0.0064085, 0.0052565, 0.0064071, -0.0011532, 0.0011520
6: -0.0020780, -0.0010285, -0.0020767, -0.0010308, -0.0010472, 0.0010482
7: -0.0082324, -0.0075494, -0.0082312, -0.0075505, -0.0006819, 0.0006818
8: 0.0054717, 0.0094441, 0.0054803, 0.0094392, -0.0039122, 0.0039034
9: -0.0036828, -0.0032159, -0.0036827, -0.0032174, -0.0004654, 0.0004669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006295
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006295
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007886, 0.0011153, 0.0007728, 0.0011157, -0.0003272, 0.0003425
1: 0.9934304, 0.9942093, 0.9934254, 0.9942425, -0.0008121, 0.0007839
2: -0.0085521, -0.0055060, -0.0086064, -0.0056463, -0.0026804, 0.0028752
3: 0.0036746, 0.0041459, 0.0036550, 0.0041495, -0.0004748, 0.0004909
4: 0.0027686, 0.0051761, 0.0028796, 0.0052190, -0.0024504, 0.0022966
5: 0.0052310, 0.0064062, 0.0051937, 0.0063759, -0.0011449, 0.0012125
6: -0.0020896, -0.0010100, -0.0021084, -0.0009756, -0.0011140, 0.0010985
7: -0.0082304, -0.0075396, -0.0082039, -0.0075237, -0.0007067, 0.0006642
8: 0.0054858, 0.0094881, 0.0056702, 0.0095594, -0.0040133, 0.0037570
9: -0.0036827, -0.0032018, -0.0036813, -0.0031791, -0.0005036, 0.0004794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006043, upper bound: 0.0005979
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006355, upper bound: 0.0006228
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007811, 0.0011155, 0.0007711, 0.0011158, -0.0003347, 0.0003445
1: 0.9934281, 0.9942253, 0.9934247, 0.9942464, -0.0008184, 0.0008005
2: -0.0085780, -0.0054868, -0.0086126, -0.0056443, -0.0027124, 0.0029078
3: 0.0036652, 0.0041476, 0.0036527, 0.0041499, -0.0004846, 0.0004949
4: 0.0027535, 0.0051966, 0.0028780, 0.0052240, -0.0024705, 0.0023186
5: 0.0052132, 0.0064104, 0.0051895, 0.0063763, -0.0011631, 0.0012209
6: -0.0020986, -0.0009936, -0.0021106, -0.0009716, -0.0011270, 0.0011170
7: -0.0082340, -0.0075320, -0.0082042, -0.0075218, -0.0007122, 0.0006722
8: 0.0054606, 0.0095221, 0.0056676, 0.0095676, -0.0040475, 0.0037943
9: -0.0036829, -0.0031910, -0.0036813, -0.0031765, -0.0005064, 0.0004903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006086, upper bound: 0.0006032
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006428, upper bound: 0.0006322
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007968, 0.0011151, 0.0008012, 0.0011150, -0.0003182, 0.0003139
1: 0.9934333, 0.9941917, 0.9934346, 0.9941824, -0.0007491, 0.0007571
2: -0.0085235, -0.0055075, -0.0085083, -0.0055092, -0.0027889, 0.0027903
3: 0.0036850, 0.0041440, 0.0036905, 0.0041430, -0.0004580, 0.0004535
4: 0.0027698, 0.0051536, 0.0027712, 0.0051415, -0.0023717, 0.0023824
5: 0.0052506, 0.0064059, 0.0052610, 0.0064055, -0.0011550, 0.0011449
6: -0.0020797, -0.0010281, -0.0020744, -0.0010334, -0.0010463, 0.0010463
7: -0.0082301, -0.0075480, -0.0082298, -0.0075525, -0.0006777, 0.0006818
8: 0.0054877, 0.0094505, 0.0054900, 0.0094305, -0.0038863, 0.0038999
9: -0.0036827, -0.0032138, -0.0036827, -0.0032202, -0.0004625, 0.0004689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006077, upper bound: 0.0006146
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006077, upper bound: 0.0006146
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007893, 0.0011153, 0.0007994, 0.0011151, -0.0003258, 0.0003159
1: 0.9934308, 0.9942078, 0.9934340, 0.9941864, -0.0007556, 0.0007738
2: -0.0085496, -0.0054884, -0.0085148, -0.0055072, -0.0028209, 0.0028229
3: 0.0036755, 0.0041457, 0.0036881, 0.0041434, -0.0004678, 0.0004576
4: 0.0027548, 0.0051742, 0.0027696, 0.0051467, -0.0023919, 0.0024046
5: 0.0052327, 0.0064100, 0.0052566, 0.0064060, -0.0011733, 0.0011535
6: -0.0020887, -0.0010116, -0.0020767, -0.0010327, -0.0010561, 0.0010651
7: -0.0082337, -0.0075404, -0.0082302, -0.0075506, -0.0006832, 0.0006898
8: 0.0054627, 0.0094848, 0.0054874, 0.0094390, -0.0039211, 0.0039382
9: -0.0036829, -0.0032029, -0.0036827, -0.0032175, -0.0004654, 0.0004798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005668
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005668
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007906, 0.0011153, 0.0007970, 0.0011151, -0.0003245, 0.0003183
1: 0.9934312, 0.9942049, 0.9934332, 0.9941916, -0.0007604, 0.0007717
2: -0.0085451, -0.0054894, -0.0085231, -0.0054652, -0.0028455, 0.0028018
3: 0.0036772, 0.0041454, 0.0036851, 0.0041439, -0.0004668, 0.0004603
4: 0.0027555, 0.0051706, 0.0027364, 0.0051532, -0.0023977, 0.0024342
5: 0.0052358, 0.0064098, 0.0052509, 0.0064150, -0.0011792, 0.0011589
6: -0.0020872, -0.0010145, -0.0020795, -0.0010181, -0.0010691, 0.0010651
7: -0.0082335, -0.0075417, -0.0082381, -0.0075481, -0.0006854, 0.0006964
8: 0.0054640, 0.0094788, 0.0054322, 0.0094499, -0.0039240, 0.0039837
9: -0.0036829, -0.0032048, -0.0036831, -0.0032140, -0.0004689, 0.0004783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006472
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006472
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0008194, 0.0011145, 0.0008052, 0.0011149, -0.0002955, 0.0003093
1: 0.9934405, 0.9941441, 0.9934359, 0.9941740, -0.0007335, 0.0007082
2: -0.0084456, -0.0053585, -0.0084945, -0.0054668, -0.0027611, 0.0029048
3: 0.0037132, 0.0041388, 0.0036955, 0.0041420, -0.0004288, 0.0004433
4: 0.0026521, 0.0050920, 0.0027376, 0.0051306, -0.0024785, 0.0023544
5: 0.0053040, 0.0064381, 0.0052705, 0.0064147, -0.0011107, 0.0011676
6: -0.0020526, -0.0009811, -0.0020696, -0.0010186, -0.0010340, 0.0010885
7: -0.0082583, -0.0075709, -0.0082378, -0.0075565, -0.0007017, 0.0006670
8: 0.0052920, 0.0093482, 0.0054342, 0.0094124, -0.0040577, 0.0038555
9: -0.0036842, -0.0032465, -0.0036831, -0.0032260, -0.0004582, 0.0004366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006472
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006472
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007886, 0.0011153, 0.0007897, 0.0011153, -0.0003267, 0.0003257
1: 0.9934305, 0.9942093, 0.9934309, 0.9942070, -0.0007765, 0.0007784
2: -0.0085521, -0.0054873, -0.0085482, -0.0054474, -0.0028754, 0.0028280
3: 0.0036747, 0.0041459, 0.0036761, 0.0041456, -0.0004710, 0.0004698
4: 0.0027539, 0.0051761, 0.0027223, 0.0051731, -0.0024192, 0.0024538
5: 0.0052310, 0.0064103, 0.0052336, 0.0064189, -0.0011879, 0.0011766
6: -0.0020896, -0.0010100, -0.0020883, -0.0010119, -0.0010777, 0.0010783
7: -0.0082339, -0.0075396, -0.0082415, -0.0075408, -0.0006932, 0.0007018
8: 0.0054613, 0.0094880, 0.0054088, 0.0094830, -0.0039595, 0.0040175
9: -0.0036829, -0.0032018, -0.0036833, -0.0032035, -0.0004794, 0.0004815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006477
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006477
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0008173, 0.0011146, 0.0007981, 0.0011151, -0.0002978, 0.0003164
1: 0.9934399, 0.9941484, 0.9934337, 0.9941890, -0.0007492, 0.0007147
2: -0.0084528, -0.0053562, -0.0085190, -0.0054491, -0.0027888, 0.0029298
3: 0.0037106, 0.0041393, 0.0036866, 0.0041437, -0.0004331, 0.0004527
4: 0.0026503, 0.0050977, 0.0027237, 0.0051500, -0.0024998, 0.0023740
5: 0.0052991, 0.0064386, 0.0052537, 0.0064185, -0.0011194, 0.0011849
6: -0.0020551, -0.0009803, -0.0020781, -0.0010125, -0.0010426, 0.0010979
7: -0.0082587, -0.0075688, -0.0082412, -0.0075493, -0.0007094, 0.0006724
8: 0.0052890, 0.0093576, 0.0054111, 0.0094446, -0.0040930, 0.0038890
9: -0.0036842, -0.0032434, -0.0036833, -0.0032157, -0.0004685, 0.0004399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006477
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006477
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007824, 0.0011155, 0.0007970, 0.0011151, -0.0003327, 0.0003185
1: 0.9934285, 0.9942223, 0.9934332, 0.9941913, -0.0007628, 0.0007891
2: -0.0085734, -0.0054826, -0.0085229, -0.0054691, -0.0028732, 0.0028095
3: 0.0036669, 0.0041473, 0.0036852, 0.0041439, -0.0004770, 0.0004621
4: 0.0027502, 0.0051930, 0.0027395, 0.0051531, -0.0024029, 0.0024535
5: 0.0052163, 0.0064113, 0.0052510, 0.0064142, -0.0011978, 0.0011603
6: -0.0020970, -0.0009965, -0.0020795, -0.0010195, -0.0010775, 0.0010830
7: -0.0082348, -0.0075334, -0.0082374, -0.0075482, -0.0006866, 0.0007040
8: 0.0054551, 0.0095161, 0.0054374, 0.0094497, -0.0039326, 0.0040162
9: -0.0036829, -0.0031929, -0.0036831, -0.0032141, -0.0004689, 0.0004902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006432
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006432
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0008104, 0.0011148, 0.0008053, 0.0011149, -0.0003045, 0.0003095
1: 0.9934376, 0.9941629, 0.9934360, 0.9941739, -0.0007363, 0.0007269
2: -0.0084765, -0.0053468, -0.0084943, -0.0054707, -0.0027909, 0.0029183
3: 0.0037020, 0.0041409, 0.0036955, 0.0041420, -0.0004400, 0.0004453
4: 0.0026429, 0.0051164, 0.0027408, 0.0051305, -0.0024876, 0.0023757
5: 0.0052828, 0.0064406, 0.0052706, 0.0064139, -0.0011310, 0.0011700
6: -0.0020634, -0.0009770, -0.0020696, -0.0010200, -0.0010434, 0.0010925
7: -0.0082605, -0.0075618, -0.0082371, -0.0075566, -0.0007039, 0.0006753
8: 0.0052767, 0.0093888, 0.0054394, 0.0094122, -0.0040743, 0.0038911
9: -0.0036843, -0.0032335, -0.0036831, -0.0032261, -0.0004583, 0.0004496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006333, upper bound: 0.0006397
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006432
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007800, 0.0011156, 0.0007897, 0.0011153, -0.0003353, 0.0003258
1: 0.9934277, 0.9942275, 0.9934309, 0.9942068, -0.0007792, 0.0007967
2: -0.0085818, -0.0054805, -0.0085481, -0.0054508, -0.0029028, 0.0028358
3: 0.0036639, 0.0041478, 0.0036761, 0.0041456, -0.0004817, 0.0004717
4: 0.0027485, 0.0051996, 0.0027251, 0.0051730, -0.0024244, 0.0024745
5: 0.0052106, 0.0064117, 0.0052337, 0.0064181, -0.0012075, 0.0011780
6: -0.0020999, -0.0009912, -0.0020882, -0.0010126, -0.0010873, 0.0010970
7: -0.0082352, -0.0075309, -0.0082408, -0.0075408, -0.0006944, 0.0007099
8: 0.0054524, 0.0095270, 0.0054133, 0.0094828, -0.0039682, 0.0040525
9: -0.0036830, -0.0031894, -0.0036833, -0.0032035, -0.0004794, 0.0004939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006432
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006432
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0008084, 0.0011148, 0.0007982, 0.0011151, -0.0003067, 0.0003166
1: 0.9934369, 0.9941673, 0.9934337, 0.9941889, -0.0007520, 0.0007337
2: -0.0084837, -0.0053449, -0.0085189, -0.0054525, -0.0028218, 0.0029438
3: 0.0036994, 0.0041413, 0.0036867, 0.0041437, -0.0004443, 0.0004547
4: 0.0026413, 0.0051221, 0.0027264, 0.0051499, -0.0025086, 0.0023957
5: 0.0052779, 0.0064411, 0.0052537, 0.0064178, -0.0011399, 0.0011873
6: -0.0020659, -0.0009763, -0.0020781, -0.0010137, -0.0010522, 0.0011017
7: -0.0082609, -0.0075597, -0.0082405, -0.0075494, -0.0007115, 0.0006808
8: 0.0052741, 0.0093982, 0.0054155, 0.0094444, -0.0041089, 0.0039259
9: -0.0036844, -0.0032305, -0.0036832, -0.0032157, -0.0004686, 0.0004528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006319, upper bound: 0.0006398
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006432
time: 0.70 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.08 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006256, upper bound: 0.0006354
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006256, upper bound: 0.0006354
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006256, upper bound: 0.0006354
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006256, upper bound: 0.0006354
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006351, upper bound: 0.0006429
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006351, upper bound: 0.0006429
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006351, upper bound: 0.0006429
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006351, upper bound: 0.0006429
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0005945, upper bound: 0.0005935
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0005945, upper bound: 0.0006078
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0005945, upper bound: 0.0005935
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0005945, upper bound: 0.0005935
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006265, upper bound: 0.0006083
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006265, upper bound: 0.0006083
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006265, upper bound: 0.0006083
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006265, upper bound: 0.0006083
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0005979, upper bound: 0.0006043
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0005979, upper bound: 0.0006043
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006228, upper bound: 0.0006355
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006228, upper bound: 0.0006355
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006032, upper bound: 0.0006086
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006032, upper bound: 0.0006086
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006321, upper bound: 0.0006428
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006321, upper bound: 0.0006428
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005550
IS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005550
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005550
IS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005550
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0005362, upper bound: 0.0006083
IS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0005362, upper bound: 0.0005362
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006354, upper bound: 0.0006256
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006354, upper bound: 0.0006256
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006429, upper bound: 0.0006351
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006429, upper bound: 0.0006351
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006078, upper bound: 0.0006160
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006078, upper bound: 0.0006160
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006295
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006295
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006043, upper bound: 0.0005979
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006355, upper bound: 0.0006228
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006086, upper bound: 0.0006032
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006428, upper bound: 0.0006322
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006077, upper bound: 0.0006146
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006077, upper bound: 0.0006146
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005668
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005668
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006472
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006472
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006472
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006472
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006477
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006477
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006477
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006477
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006432
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006432
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006333, upper bound: 0.0006397
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006432
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006432
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006432
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006319, upper bound: 0.0006398
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 1, lower bound: -0.0006432, upper bound: 0.0006432

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007729, 0.0011157, 0.0008065, 0.0011149, -0.0003420, 0.0003092
1: 0.9934254, 0.9942424, 0.9934363, 0.9941713, -0.0007460, 0.0008061
2: -0.0086062, -0.0056797, -0.0084901, -0.0055428, -0.0028378, 0.0025821
3: 0.0036550, 0.0041494, 0.0036971, 0.0041418, -0.0004867, 0.0004524
4: 0.0029060, 0.0052189, 0.0027978, 0.0051271, -0.0022211, 0.0024211
5: 0.0051939, 0.0063687, 0.0052735, 0.0063983, -0.0012044, 0.0010951
6: -0.0021084, -0.0009757, -0.0020681, -0.0010451, -0.0010633, 0.0010924
7: -0.0081976, -0.0075238, -0.0082234, -0.0075578, -0.0006397, 0.0006996
8: 0.0057140, 0.0095591, 0.0055342, 0.0094066, -0.0036315, 0.0039642
9: -0.0036809, -0.0031792, -0.0036823, -0.0032278, -0.0004531, 0.0005032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006230, upper bound: 0.0006366
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006419, upper bound: 0.0006630
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007729, 0.0011157, 0.0008352, 0.0011141, -0.0003412, 0.0002806
1: 0.9934254, 0.9942424, 0.9934458, 0.9941105, -0.0006852, 0.0007966
2: -0.0086062, -0.0056797, -0.0083911, -0.0054109, -0.0029691, 0.0024832
3: 0.0036550, 0.0041494, 0.0037329, 0.0041352, -0.0004802, 0.0004165
4: 0.0029060, 0.0052189, 0.0026935, 0.0050489, -0.0021429, 0.0025254
5: 0.0051939, 0.0063687, 0.0053414, 0.0064268, -0.0012329, 0.0010273
6: -0.0021084, -0.0009757, -0.0020337, -0.0009993, -0.0011091, 0.0010580
7: -0.0081976, -0.0075238, -0.0082484, -0.0075869, -0.0006107, 0.0007215
8: 0.0057140, 0.0095591, 0.0053608, 0.0092766, -0.0035014, 0.0041378
9: -0.0036809, -0.0031792, -0.0036837, -0.0032693, -0.0004116, 0.0005045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006230, upper bound: 0.0006366
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006419, upper bound: 0.0006630
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007669, 0.0011159, 0.0008065, 0.0011149, -0.0003480, 0.0003094
1: 0.9934233, 0.9942552, 0.9934363, 0.9941713, -0.0007480, 0.0008188
2: -0.0086270, -0.0056870, -0.0084901, -0.0055428, -0.0028628, 0.0025774
3: 0.0036475, 0.0041508, 0.0036971, 0.0041418, -0.0004942, 0.0004537
4: 0.0029117, 0.0052353, 0.0027978, 0.0051271, -0.0022154, 0.0024376
5: 0.0051796, 0.0063671, 0.0052735, 0.0063983, -0.0012187, 0.0010936
6: -0.0021156, -0.0009625, -0.0020681, -0.0010451, -0.0010706, 0.0011056
7: -0.0081962, -0.0075177, -0.0082234, -0.0075578, -0.0006384, 0.0007053
8: 0.0057236, 0.0095865, 0.0055342, 0.0094066, -0.0036227, 0.0039922
9: -0.0036808, -0.0031704, -0.0036823, -0.0032278, -0.0004530, 0.0005119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005982, upper bound: 0.0006037
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006256, upper bound: 0.0006354
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007669, 0.0011159, 0.0008352, 0.0011141, -0.0003472, 0.0002807
1: 0.9934233, 0.9942552, 0.9934458, 0.9941105, -0.0006872, 0.0008094
2: -0.0086270, -0.0056870, -0.0083911, -0.0054109, -0.0029941, 0.0024784
3: 0.0036475, 0.0041508, 0.0037329, 0.0041352, -0.0004877, 0.0004179
4: 0.0029117, 0.0052353, 0.0026935, 0.0050489, -0.0021372, 0.0025419
5: 0.0051796, 0.0063671, 0.0053414, 0.0064268, -0.0012472, 0.0010257
6: -0.0021156, -0.0009625, -0.0020337, -0.0009993, -0.0011163, 0.0010713
7: -0.0081962, -0.0075177, -0.0082484, -0.0075869, -0.0006093, 0.0007273
8: 0.0057236, 0.0095865, 0.0053608, 0.0092766, -0.0034926, 0.0041659
9: -0.0036808, -0.0031704, -0.0036837, -0.0032693, -0.0004116, 0.0005132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005982, upper bound: 0.0006037
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006256, upper bound: 0.0006354
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007711, 0.0011158, 0.0007991, 0.0011151, -0.0003440, 0.0003167
1: 0.9934248, 0.9942463, 0.9934340, 0.9941870, -0.0007622, 0.0008124
2: -0.0086124, -0.0056777, -0.0085158, -0.0055232, -0.0028639, 0.0026110
3: 0.0036528, 0.0041499, 0.0036878, 0.0041435, -0.0004907, 0.0004621
4: 0.0029044, 0.0052238, 0.0027823, 0.0051475, -0.0022431, 0.0024416
5: 0.0051896, 0.0063691, 0.0052559, 0.0064025, -0.0012129, 0.0011132
6: -0.0021105, -0.0009717, -0.0020770, -0.0010330, -0.0010775, 0.0011053
7: -0.0081979, -0.0075219, -0.0082271, -0.0075503, -0.0006477, 0.0007052
8: 0.0057114, 0.0095673, 0.0055084, 0.0094404, -0.0036681, 0.0039978
9: -0.0036809, -0.0031766, -0.0036825, -0.0032170, -0.0004639, 0.0005060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006245, upper bound: 0.0006374
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006465, upper bound: 0.0006668
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007711, 0.0011158, 0.0008285, 0.0011143, -0.0003432, 0.0002873
1: 0.9934248, 0.9942463, 0.9934436, 0.9941247, -0.0006999, 0.0008027
2: -0.0086124, -0.0056777, -0.0084142, -0.0053909, -0.0029953, 0.0025117
3: 0.0036528, 0.0041499, 0.0037246, 0.0041367, -0.0004839, 0.0004253
4: 0.0029044, 0.0052238, 0.0026777, 0.0050672, -0.0021628, 0.0025461
5: 0.0051896, 0.0063691, 0.0053256, 0.0064311, -0.0012415, 0.0010435
6: -0.0021105, -0.0009717, -0.0020417, -0.0009923, -0.0011182, 0.0010700
7: -0.0081979, -0.0075219, -0.0082522, -0.0075801, -0.0006179, 0.0007289
8: 0.0057114, 0.0095673, 0.0053346, 0.0093069, -0.0035349, 0.0041720
9: -0.0036809, -0.0031766, -0.0036839, -0.0032596, -0.0004213, 0.0005073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006245, upper bound: 0.0006374
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006465, upper bound: 0.0006668
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007651, 0.0011159, 0.0007991, 0.0011151, -0.0003499, 0.0003169
1: 0.9934227, 0.9942590, 0.9934340, 0.9941870, -0.0007643, 0.0008250
2: -0.0086331, -0.0056850, -0.0085158, -0.0055232, -0.0028881, 0.0026063
3: 0.0036453, 0.0041512, 0.0036878, 0.0041435, -0.0004981, 0.0004634
4: 0.0029101, 0.0052401, 0.0027823, 0.0051475, -0.0022374, 0.0024579
5: 0.0051755, 0.0063675, 0.0052559, 0.0064025, -0.0012270, 0.0011117
6: -0.0021177, -0.0009586, -0.0020770, -0.0010330, -0.0010847, 0.0011184
7: -0.0081966, -0.0075159, -0.0082271, -0.0075503, -0.0006463, 0.0007113
8: 0.0057209, 0.0095944, 0.0055084, 0.0094404, -0.0036594, 0.0040259
9: -0.0036809, -0.0031679, -0.0036825, -0.0032170, -0.0004638, 0.0005146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006034, upper bound: 0.0006080
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006351, upper bound: 0.0006429
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007651, 0.0011159, 0.0008285, 0.0011143, -0.0003492, 0.0002875
1: 0.9934227, 0.9942590, 0.9934436, 0.9941247, -0.0007020, 0.0008154
2: -0.0086331, -0.0056850, -0.0084142, -0.0053909, -0.0030195, 0.0025069
3: 0.0036453, 0.0041512, 0.0037246, 0.0041367, -0.0004914, 0.0004267
4: 0.0029101, 0.0052401, 0.0026777, 0.0050672, -0.0021571, 0.0025624
5: 0.0051755, 0.0063675, 0.0053256, 0.0064311, -0.0012556, 0.0010420
6: -0.0021177, -0.0009586, -0.0020417, -0.0009923, -0.0011254, 0.0010831
7: -0.0081966, -0.0075159, -0.0082522, -0.0075801, -0.0006165, 0.0007347
8: 0.0057209, 0.0095944, 0.0053346, 0.0093069, -0.0035262, 0.0042001
9: -0.0036809, -0.0031679, -0.0036839, -0.0032596, -0.0004213, 0.0005159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006034, upper bound: 0.0006080
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006351, upper bound: 0.0006428
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0008013, 0.0011150, 0.0007885, 0.0011153, -0.0003140, 0.0003265
1: 0.9934347, 0.9941822, 0.9934304, 0.9942093, -0.0007746, 0.0007517
2: -0.0085080, -0.0055472, -0.0085523, -0.0057032, -0.0025918, 0.0027768
3: 0.0036906, 0.0041429, 0.0036746, 0.0041459, -0.0004553, 0.0004684
4: 0.0028013, 0.0051413, 0.0029245, 0.0051763, -0.0023750, 0.0022168
5: 0.0052612, 0.0063973, 0.0052309, 0.0063636, -0.0011024, 0.0011664
6: -0.0020743, -0.0010379, -0.0020897, -0.0010099, -0.0010644, 0.0010517
7: -0.0082226, -0.0075526, -0.0081931, -0.0075396, -0.0006830, 0.0006405
8: 0.0055400, 0.0094302, 0.0057450, 0.0094883, -0.0038873, 0.0036282
9: -0.0036823, -0.0032203, -0.0036807, -0.0032018, -0.0004805, 0.0004604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006054
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006241
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0008013, 0.0011150, 0.0008054, 0.0011149, -0.0003136, 0.0003096
1: 0.9934347, 0.9941822, 0.9934360, 0.9941737, -0.0007390, 0.0007461
2: -0.0085080, -0.0055472, -0.0084941, -0.0055135, -0.0027835, 0.0027201
3: 0.0036906, 0.0041429, 0.0036957, 0.0041420, -0.0004514, 0.0004473
4: 0.0028013, 0.0051413, 0.0027746, 0.0051303, -0.0023290, 0.0023668
5: 0.0052612, 0.0063973, 0.0052708, 0.0064046, -0.0011434, 0.0011265
6: -0.0020743, -0.0010379, -0.0020695, -0.0010349, -0.0010395, 0.0010315
7: -0.0082226, -0.0075526, -0.0082290, -0.0075567, -0.0006659, 0.0006764
8: 0.0055400, 0.0094302, 0.0054957, 0.0094118, -0.0038113, 0.0038779
9: -0.0036823, -0.0032203, -0.0036826, -0.0032262, -0.0004561, 0.0004623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006054
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006241
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007947, 0.0011152, 0.0007885, 0.0011153, -0.0003206, 0.0003267
1: 0.9934325, 0.9941963, 0.9934304, 0.9942093, -0.0007768, 0.0007659
2: -0.0085309, -0.0055494, -0.0085523, -0.0057032, -0.0026177, 0.0027744
3: 0.0036823, 0.0041445, 0.0036746, 0.0041459, -0.0004636, 0.0004699
4: 0.0028030, 0.0051594, 0.0029245, 0.0051763, -0.0023733, 0.0022348
5: 0.0052455, 0.0063968, 0.0052309, 0.0063636, -0.0011181, 0.0011660
6: -0.0020822, -0.0010235, -0.0020897, -0.0010099, -0.0010723, 0.0010662
7: -0.0082222, -0.0075459, -0.0081931, -0.0075396, -0.0006826, 0.0006473
8: 0.0055428, 0.0094602, 0.0057450, 0.0094883, -0.0038843, 0.0036591
9: -0.0036823, -0.0032107, -0.0036807, -0.0032018, -0.0004805, 0.0004699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005534, upper bound: 0.0005491
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005534, upper bound: 0.0005491
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007947, 0.0011152, 0.0008054, 0.0011149, -0.0003202, 0.0003098
1: 0.9934325, 0.9941963, 0.9934360, 0.9941737, -0.0007412, 0.0007603
2: -0.0085309, -0.0055494, -0.0084941, -0.0055135, -0.0028094, 0.0027177
3: 0.0036823, 0.0041445, 0.0036957, 0.0041420, -0.0004597, 0.0004488
4: 0.0028030, 0.0051594, 0.0027746, 0.0051303, -0.0023273, 0.0023848
5: 0.0052455, 0.0063968, 0.0052708, 0.0064046, -0.0011591, 0.0011260
6: -0.0020822, -0.0010235, -0.0020695, -0.0010349, -0.0010474, 0.0010460
7: -0.0082222, -0.0075459, -0.0082290, -0.0075567, -0.0006655, 0.0006831
8: 0.0055428, 0.0094602, 0.0054957, 0.0094118, -0.0038083, 0.0039088
9: -0.0036823, -0.0032107, -0.0036826, -0.0032262, -0.0004561, 0.0004719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005534, upper bound: 0.0005491
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005534, upper bound: 0.0005491
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007994, 0.0011151, 0.0007824, 0.0011155, -0.0003161, 0.0003327
1: 0.9934340, 0.9941863, 0.9934284, 0.9942224, -0.0007884, 0.0007579
2: -0.0085146, -0.0055453, -0.0085735, -0.0056865, -0.0026136, 0.0027969
3: 0.0036882, 0.0041434, 0.0036669, 0.0041473, -0.0004590, 0.0004765
4: 0.0027997, 0.0051465, 0.0029113, 0.0051930, -0.0023933, 0.0022352
5: 0.0052567, 0.0063977, 0.0052163, 0.0063672, -0.0011105, 0.0011814
6: -0.0020766, -0.0010338, -0.0020970, -0.0009965, -0.0010801, 0.0010632
7: -0.0082230, -0.0075507, -0.0081963, -0.0075334, -0.0006896, 0.0006456
8: 0.0055374, 0.0094387, 0.0057229, 0.0095161, -0.0039168, 0.0036584
9: -0.0036823, -0.0032176, -0.0036809, -0.0031929, -0.0004894, 0.0004633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006054
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006367
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007994, 0.0011151, 0.0007982, 0.0011151, -0.0003156, 0.0003168
1: 0.9934340, 0.9941863, 0.9934337, 0.9941887, -0.0007547, 0.0007526
2: -0.0085146, -0.0055453, -0.0085186, -0.0054952, -0.0028124, 0.0027478
3: 0.0036882, 0.0041434, 0.0036867, 0.0041436, -0.0004554, 0.0004566
4: 0.0027997, 0.0051465, 0.0027601, 0.0051497, -0.0023500, 0.0023863
5: 0.0052567, 0.0063977, 0.0052539, 0.0064085, -0.0011518, 0.0011438
6: -0.0020766, -0.0010338, -0.0020780, -0.0010285, -0.0010480, 0.0010442
7: -0.0082230, -0.0075507, -0.0082324, -0.0075494, -0.0006735, 0.0006818
8: 0.0055374, 0.0094387, 0.0054717, 0.0094441, -0.0038465, 0.0039118
9: -0.0036823, -0.0032176, -0.0036828, -0.0032159, -0.0004664, 0.0004652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006054
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006367
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007930, 0.0011152, 0.0007824, 0.0011155, -0.0003225, 0.0003328
1: 0.9934319, 0.9942000, 0.9934284, 0.9942224, -0.0007905, 0.0007716
2: -0.0085369, -0.0055474, -0.0085735, -0.0056865, -0.0026392, 0.0027943
3: 0.0036801, 0.0041449, 0.0036669, 0.0041473, -0.0004672, 0.0004780
4: 0.0028014, 0.0051641, 0.0029113, 0.0051930, -0.0023917, 0.0022529
5: 0.0052414, 0.0063973, 0.0052163, 0.0063672, -0.0011258, 0.0011810
6: -0.0020843, -0.0010196, -0.0020970, -0.0009965, -0.0010879, 0.0010774
7: -0.0082226, -0.0075441, -0.0081963, -0.0075334, -0.0006892, 0.0006522
8: 0.0055402, 0.0094681, 0.0057229, 0.0095161, -0.0039139, 0.0036887
9: -0.0036823, -0.0032082, -0.0036809, -0.0031929, -0.0004894, 0.0004726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005534, upper bound: 0.0005491
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005534, upper bound: 0.0005491
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007930, 0.0011152, 0.0007982, 0.0011151, -0.0003221, 0.0003170
1: 0.9934319, 0.9942000, 0.9934337, 0.9941887, -0.0007568, 0.0007663
2: -0.0085369, -0.0055474, -0.0085186, -0.0054952, -0.0028379, 0.0027453
3: 0.0036801, 0.0041449, 0.0036867, 0.0041436, -0.0004635, 0.0004581
4: 0.0028014, 0.0051641, 0.0027601, 0.0051497, -0.0023483, 0.0024040
5: 0.0052414, 0.0063973, 0.0052539, 0.0064085, -0.0011672, 0.0011433
6: -0.0020843, -0.0010196, -0.0020780, -0.0010285, -0.0010558, 0.0010583
7: -0.0082226, -0.0075441, -0.0082324, -0.0075494, -0.0006731, 0.0006884
8: 0.0055402, 0.0094681, 0.0054717, 0.0094441, -0.0038435, 0.0039420
9: -0.0036823, -0.0032082, -0.0036828, -0.0032159, -0.0004664, 0.0004746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005534, upper bound: 0.0005491
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005534, upper bound: 0.0006083
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007677, 0.0011159, 0.0007980, 0.0011151, -0.0003474, 0.0003178
1: 0.9934236, 0.9942535, 0.9934336, 0.9941893, -0.0007657, 0.0008199
2: -0.0086242, -0.0057134, -0.0085194, -0.0055538, -0.0028482, 0.0025840
3: 0.0036485, 0.0041506, 0.0036865, 0.0041437, -0.0004952, 0.0004641
4: 0.0029325, 0.0052331, 0.0028064, 0.0051503, -0.0022177, 0.0024267
5: 0.0051815, 0.0063614, 0.0052534, 0.0063959, -0.0012144, 0.0011080
6: -0.0021146, -0.0009643, -0.0020782, -0.0010308, -0.0010839, 0.0011140
7: -0.0081912, -0.0075185, -0.0082214, -0.0075492, -0.0006420, 0.0006954
8: 0.0057583, 0.0095828, 0.0055486, 0.0094451, -0.0036272, 0.0039745
9: -0.0036806, -0.0031716, -0.0036822, -0.0032155, -0.0004650, 0.0005106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005836, upper bound: 0.0005852
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005836, upper bound: 0.0005852
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007677, 0.0011159, 0.0008263, 0.0011143, -0.0003467, 0.0002896
1: 0.9934236, 0.9942535, 0.9934430, 0.9941294, -0.0007058, 0.0008106
2: -0.0086242, -0.0057134, -0.0084218, -0.0054207, -0.0029806, 0.0024894
3: 0.0036485, 0.0041506, 0.0037218, 0.0041372, -0.0004887, 0.0004288
4: 0.0029325, 0.0052331, 0.0027012, 0.0050732, -0.0021406, 0.0025319
5: 0.0051815, 0.0063614, 0.0053204, 0.0064247, -0.0012432, 0.0010410
6: -0.0021146, -0.0009643, -0.0020444, -0.0010027, -0.0011120, 0.0010801
7: -0.0081912, -0.0075185, -0.0082465, -0.0075779, -0.0006133, 0.0007175
8: 0.0057583, 0.0095828, 0.0053738, 0.0093169, -0.0034989, 0.0041493
9: -0.0036806, -0.0031716, -0.0036836, -0.0032564, -0.0004241, 0.0005119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005836, upper bound: 0.0005852
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005836, upper bound: 0.0006043
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007736, 0.0011157, 0.0007980, 0.0011151, -0.0003415, 0.0003178
1: 0.9934255, 0.9942411, 0.9934335, 0.9941894, -0.0007639, 0.0008076
2: -0.0086039, -0.0056800, -0.0085197, -0.0055394, -0.0028394, 0.0026154
3: 0.0036559, 0.0041493, 0.0036864, 0.0041437, -0.0004878, 0.0004629
4: 0.0029062, 0.0052171, 0.0027950, 0.0051505, -0.0022443, 0.0024221
5: 0.0051954, 0.0063686, 0.0052532, 0.0063990, -0.0012036, 0.0011154
6: -0.0021076, -0.0009771, -0.0020783, -0.0010306, -0.0010770, 0.0011012
7: -0.0081975, -0.0075244, -0.0082241, -0.0075491, -0.0006484, 0.0006950
8: 0.0057145, 0.0095562, 0.0055297, 0.0094455, -0.0036706, 0.0039662
9: -0.0036809, -0.0031801, -0.0036824, -0.0032154, -0.0004655, 0.0005022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006120, upper bound: 0.0006131
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006120, upper bound: 0.0006355
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007736, 0.0011157, 0.0008262, 0.0011144, -0.0003408, 0.0002895
1: 0.9934255, 0.9942411, 0.9934429, 0.9941295, -0.0007040, 0.0007982
2: -0.0086039, -0.0056800, -0.0084221, -0.0054071, -0.0029716, 0.0025219
3: 0.0036559, 0.0041493, 0.0037217, 0.0041373, -0.0004814, 0.0004276
4: 0.0029062, 0.0052171, 0.0026905, 0.0050734, -0.0021672, 0.0025266
5: 0.0051954, 0.0063686, 0.0053201, 0.0064276, -0.0012322, 0.0010485
6: -0.0021076, -0.0009771, -0.0020445, -0.0009979, -0.0011097, 0.0010674
7: -0.0081975, -0.0075244, -0.0082491, -0.0075778, -0.0006197, 0.0007170
8: 0.0057145, 0.0095562, 0.0053559, 0.0093173, -0.0035426, 0.0041401
9: -0.0036809, -0.0031801, -0.0036837, -0.0032563, -0.0004246, 0.0005036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006120, upper bound: 0.0006131
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006120, upper bound: 0.0006355
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007658, 0.0011159, 0.0007906, 0.0011153, -0.0003495, 0.0003253
1: 0.9934230, 0.9942575, 0.9934312, 0.9942050, -0.0007820, 0.0008263
2: -0.0086308, -0.0057112, -0.0085449, -0.0055413, -0.0028707, 0.0026144
3: 0.0036461, 0.0041511, 0.0036772, 0.0041454, -0.0004993, 0.0004738
4: 0.0029308, 0.0052383, 0.0027966, 0.0051705, -0.0022396, 0.0024418
5: 0.0051770, 0.0063619, 0.0052359, 0.0063986, -0.0012216, 0.0011260
6: -0.0021169, -0.0009601, -0.0020871, -0.0010146, -0.0011024, 0.0011270
7: -0.0081916, -0.0075165, -0.0082237, -0.0075417, -0.0006499, 0.0007036
8: 0.0057554, 0.0095914, 0.0055322, 0.0094786, -0.0036640, 0.0040001
9: -0.0036806, -0.0031689, -0.0036823, -0.0032049, -0.0004757, 0.0005135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005831, upper bound: 0.0005829
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005831, upper bound: 0.0006086
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007658, 0.0011159, 0.0008183, 0.0011146, -0.0003488, 0.0002976
1: 0.9934230, 0.9942575, 0.9934403, 0.9941465, -0.0007235, 0.0008172
2: -0.0086308, -0.0057112, -0.0084495, -0.0054051, -0.0030074, 0.0025207
3: 0.0036461, 0.0041511, 0.0037118, 0.0041391, -0.0004929, 0.0004393
4: 0.0029308, 0.0052383, 0.0026889, 0.0050950, -0.0021642, 0.0025494
5: 0.0051770, 0.0063619, 0.0053014, 0.0064280, -0.0012510, 0.0010605
6: -0.0021169, -0.0009601, -0.0020540, -0.0009972, -0.0011197, 0.0010939
7: -0.0081916, -0.0075165, -0.0082495, -0.0075697, -0.0006219, 0.0007258
8: 0.0057554, 0.0095914, 0.0053532, 0.0093532, -0.0035394, 0.0041785
9: -0.0036806, -0.0031689, -0.0036837, -0.0032449, -0.0004357, 0.0005149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005831, upper bound: 0.0005829
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005831, upper bound: 0.0006086
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007718, 0.0011158, 0.0007906, 0.0011153, -0.0003435, 0.0003252
1: 0.9934250, 0.9942449, 0.9934311, 0.9942052, -0.0007802, 0.0008138
2: -0.0086101, -0.0056780, -0.0085453, -0.0055264, -0.0028629, 0.0026466
3: 0.0036536, 0.0041497, 0.0036771, 0.0041454, -0.0004918, 0.0004726
4: 0.0029046, 0.0052220, 0.0027847, 0.0051707, -0.0022661, 0.0024373
5: 0.0051912, 0.0063690, 0.0052357, 0.0064018, -0.0012106, 0.0011334
6: -0.0021098, -0.0009732, -0.0020872, -0.0010144, -0.0010954, 0.0011140
7: -0.0081979, -0.0075226, -0.0082266, -0.0075416, -0.0006562, 0.0007028
8: 0.0057118, 0.0095643, 0.0055126, 0.0094790, -0.0037072, 0.0039922
9: -0.0036809, -0.0031775, -0.0036825, -0.0032047, -0.0004762, 0.0005050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006120, upper bound: 0.0006123
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006120, upper bound: 0.0006428
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007718, 0.0011158, 0.0008182, 0.0011146, -0.0003428, 0.0002976
1: 0.9934250, 0.9942449, 0.9934402, 0.9941465, -0.0007215, 0.0008047
2: -0.0086101, -0.0056780, -0.0084497, -0.0053918, -0.0029976, 0.0025538
3: 0.0036536, 0.0041497, 0.0037117, 0.0041391, -0.0004855, 0.0004380
4: 0.0029046, 0.0052220, 0.0026783, 0.0050952, -0.0021906, 0.0025437
5: 0.0051912, 0.0063690, 0.0053012, 0.0064309, -0.0012397, 0.0010678
6: -0.0021098, -0.0009732, -0.0020541, -0.0009926, -0.0011171, 0.0010809
7: -0.0081979, -0.0075226, -0.0082520, -0.0075697, -0.0006282, 0.0007247
8: 0.0057118, 0.0095643, 0.0053357, 0.0093535, -0.0035828, 0.0041681
9: -0.0036809, -0.0031775, -0.0036839, -0.0032447, -0.0004362, 0.0005063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006120, upper bound: 0.0006122
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006120, upper bound: 0.0006428
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0008106, 0.0011148, 0.0007763, 0.0011157, -0.0003050, 0.0003385
1: 0.9934377, 0.9941624, 0.9934264, 0.9942352, -0.0007975, 0.0007360
2: -0.0084758, -0.0055152, -0.0085945, -0.0056937, -0.0025683, 0.0028505
3: 0.0037023, 0.0041408, 0.0036593, 0.0041487, -0.0004464, 0.0004815
4: 0.0027759, 0.0051158, 0.0029170, 0.0052096, -0.0024337, 0.0021988
5: 0.0052833, 0.0064042, 0.0052019, 0.0063656, -0.0010823, 0.0012023
6: -0.0020631, -0.0010355, -0.0021043, -0.0009831, -0.0010800, 0.0010689
7: -0.0082287, -0.0075620, -0.0081949, -0.0075272, -0.0007015, 0.0006329
8: 0.0054979, 0.0093878, 0.0057325, 0.0095437, -0.0039845, 0.0035983
9: -0.0036826, -0.0032338, -0.0036808, -0.0031841, -0.0004985, 0.0004470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006083
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006083
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007971, 0.0011151, 0.0007729, 0.0011157, -0.0003187, 0.0003422
1: 0.9934333, 0.9941913, 0.9934254, 0.9942424, -0.0008091, 0.0007659
2: -0.0085227, -0.0055119, -0.0086062, -0.0056797, -0.0026147, 0.0028684
3: 0.0036853, 0.0041439, 0.0036550, 0.0041494, -0.0004641, 0.0004889
4: 0.0027733, 0.0051529, 0.0029060, 0.0052189, -0.0024455, 0.0022469
5: 0.0052512, 0.0064049, 0.0051939, 0.0063687, -0.0011175, 0.0012111
6: -0.0020794, -0.0010287, -0.0021084, -0.0009757, -0.0011037, 0.0010797
7: -0.0082293, -0.0075483, -0.0081976, -0.0075238, -0.0007050, 0.0006493
8: 0.0054936, 0.0094493, 0.0057140, 0.0095591, -0.0040050, 0.0036743
9: -0.0036826, -0.0032142, -0.0036809, -0.0031792, -0.0005035, 0.0004667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006354, upper bound: 0.0006256
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006354, upper bound: 0.0006256
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007971, 0.0011151, 0.0007669, 0.0011159, -0.0003188, 0.0003482
1: 0.9934333, 0.9941913, 0.9934233, 0.9942552, -0.0008219, 0.0007679
2: -0.0085227, -0.0055119, -0.0086270, -0.0056870, -0.0026100, 0.0028933
3: 0.0036853, 0.0041439, 0.0036475, 0.0041508, -0.0004655, 0.0004964
4: 0.0027733, 0.0051529, 0.0029117, 0.0052353, -0.0024620, 0.0022411
5: 0.0052512, 0.0064049, 0.0051796, 0.0063671, -0.0011159, 0.0012254
6: -0.0020794, -0.0010287, -0.0021156, -0.0009625, -0.0011169, 0.0010869
7: -0.0082293, -0.0075483, -0.0081962, -0.0075177, -0.0007108, 0.0006479
8: 0.0054936, 0.0094493, 0.0057236, 0.0095865, -0.0040331, 0.0036655
9: -0.0036826, -0.0032142, -0.0036808, -0.0031704, -0.0005122, 0.0004667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006354, upper bound: 0.0006256
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006354, upper bound: 0.0006256
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007898, 0.0011153, 0.0007711, 0.0011158, -0.0003260, 0.0003442
1: 0.9934309, 0.9942066, 0.9934248, 0.9942463, -0.0008154, 0.0007818
2: -0.0085478, -0.0054935, -0.0086124, -0.0056777, -0.0026431, 0.0028972
3: 0.0036762, 0.0041456, 0.0036528, 0.0041499, -0.0004737, 0.0004928
4: 0.0027588, 0.0051727, 0.0029044, 0.0052238, -0.0024650, 0.0022684
5: 0.0052339, 0.0064089, 0.0051896, 0.0063691, -0.0011352, 0.0012193
6: -0.0020881, -0.0010127, -0.0021105, -0.0009717, -0.0011164, 0.0010978
7: -0.0082328, -0.0075409, -0.0081979, -0.0075219, -0.0007109, 0.0006570
8: 0.0054694, 0.0094824, 0.0057114, 0.0095673, -0.0040384, 0.0037102
9: -0.0036828, -0.0032036, -0.0036809, -0.0031766, -0.0005063, 0.0004773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006429, upper bound: 0.0006351
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006429, upper bound: 0.0006351
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007898, 0.0011153, 0.0007651, 0.0011159, -0.0003261, 0.0003502
1: 0.9934309, 0.9942066, 0.9934227, 0.9942590, -0.0008281, 0.0007839
2: -0.0085478, -0.0054935, -0.0086331, -0.0056850, -0.0026383, 0.0029214
3: 0.0036762, 0.0041456, 0.0036453, 0.0041512, -0.0004750, 0.0005003
4: 0.0027588, 0.0051727, 0.0029101, 0.0052401, -0.0024813, 0.0022626
5: 0.0052339, 0.0064089, 0.0051755, 0.0063675, -0.0011336, 0.0012335
6: -0.0020881, -0.0010127, -0.0021177, -0.0009586, -0.0011295, 0.0011050
7: -0.0082328, -0.0075409, -0.0081966, -0.0075159, -0.0007169, 0.0006557
8: 0.0054694, 0.0094824, 0.0057209, 0.0095944, -0.0040665, 0.0037015
9: -0.0036828, -0.0032036, -0.0036809, -0.0031679, -0.0005149, 0.0004772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006429, upper bound: 0.0006351
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006429, upper bound: 0.0006351
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0008054, 0.0011149, 0.0008013, 0.0011150, -0.0003096, 0.0003136
1: 0.9934360, 0.9941737, 0.9934347, 0.9941822, -0.0007461, 0.0007390
2: -0.0084941, -0.0055135, -0.0085080, -0.0055472, -0.0027201, 0.0027835
3: 0.0036957, 0.0041420, 0.0036906, 0.0041429, -0.0004473, 0.0004514
4: 0.0027746, 0.0051303, 0.0028013, 0.0051413, -0.0023668, 0.0023290
5: 0.0052708, 0.0064046, 0.0052612, 0.0063973, -0.0011265, 0.0011434
6: -0.0020695, -0.0010349, -0.0020743, -0.0010379, -0.0010315, 0.0010395
7: -0.0082290, -0.0075567, -0.0082226, -0.0075526, -0.0006764, 0.0006659
8: 0.0054957, 0.0094118, 0.0055400, 0.0094302, -0.0038779, 0.0038113
9: -0.0036826, -0.0032262, -0.0036823, -0.0032203, -0.0004623, 0.0004561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005543, upper bound: 0.0005541
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006078, upper bound: 0.0006160
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0008054, 0.0011149, 0.0007947, 0.0011152, -0.0003098, 0.0003202
1: 0.9934360, 0.9941737, 0.9934325, 0.9941963, -0.0007603, 0.0007412
2: -0.0084941, -0.0055135, -0.0085309, -0.0055494, -0.0027177, 0.0028094
3: 0.0036957, 0.0041420, 0.0036823, 0.0041445, -0.0004488, 0.0004597
4: 0.0027746, 0.0051303, 0.0028030, 0.0051594, -0.0023848, 0.0023273
5: 0.0052708, 0.0064046, 0.0052455, 0.0063968, -0.0011260, 0.0011591
6: -0.0020695, -0.0010349, -0.0020822, -0.0010235, -0.0010460, 0.0010474
7: -0.0082290, -0.0075567, -0.0082222, -0.0075459, -0.0006831, 0.0006655
8: 0.0054957, 0.0094118, 0.0055428, 0.0094602, -0.0039088, 0.0038083
9: -0.0036826, -0.0032262, -0.0036823, -0.0032107, -0.0004719, 0.0004561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005543, upper bound: 0.0005541
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006078, upper bound: 0.0006160
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007982, 0.0011151, 0.0007994, 0.0011151, -0.0003168, 0.0003156
1: 0.9934337, 0.9941887, 0.9934340, 0.9941863, -0.0007526, 0.0007547
2: -0.0085186, -0.0054952, -0.0085146, -0.0055453, -0.0027478, 0.0028124
3: 0.0036867, 0.0041436, 0.0036882, 0.0041434, -0.0004566, 0.0004554
4: 0.0027601, 0.0051497, 0.0027997, 0.0051465, -0.0023863, 0.0023500
5: 0.0052539, 0.0064085, 0.0052567, 0.0063977, -0.0011438, 0.0011518
6: -0.0020780, -0.0010285, -0.0020766, -0.0010338, -0.0010442, 0.0010480
7: -0.0082324, -0.0075494, -0.0082230, -0.0075507, -0.0006818, 0.0006735
8: 0.0054717, 0.0094441, 0.0055374, 0.0094387, -0.0039118, 0.0038465
9: -0.0036828, -0.0032159, -0.0036823, -0.0032176, -0.0004652, 0.0004664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005660, upper bound: 0.0005655
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006295
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007982, 0.0011151, 0.0007930, 0.0011152, -0.0003170, 0.0003221
1: 0.9934337, 0.9941887, 0.9934319, 0.9942000, -0.0007663, 0.0007568
2: -0.0085186, -0.0054952, -0.0085369, -0.0055474, -0.0027453, 0.0028379
3: 0.0036867, 0.0041436, 0.0036801, 0.0041449, -0.0004581, 0.0004635
4: 0.0027601, 0.0051497, 0.0028014, 0.0051641, -0.0024040, 0.0023483
5: 0.0052539, 0.0064085, 0.0052414, 0.0063973, -0.0011433, 0.0011672
6: -0.0020780, -0.0010285, -0.0020843, -0.0010196, -0.0010583, 0.0010558
7: -0.0082324, -0.0075494, -0.0082226, -0.0075441, -0.0006884, 0.0006731
8: 0.0054717, 0.0094441, 0.0055402, 0.0094681, -0.0039420, 0.0038435
9: -0.0036828, -0.0032159, -0.0036823, -0.0032082, -0.0004746, 0.0004664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005660, upper bound: 0.0005655
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006295
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007887, 0.0011153, 0.0007677, 0.0011159, -0.0003272, 0.0003476
1: 0.9934306, 0.9942090, 0.9934236, 0.9942535, -0.0008230, 0.0007854
2: -0.0085517, -0.0055226, -0.0086242, -0.0057134, -0.0026154, 0.0028796
3: 0.0036748, 0.0041458, 0.0036485, 0.0041506, -0.0004758, 0.0004973
4: 0.0027818, 0.0051758, 0.0029325, 0.0052331, -0.0024514, 0.0022433
5: 0.0052313, 0.0064026, 0.0051815, 0.0063614, -0.0011301, 0.0012211
6: -0.0020895, -0.0010103, -0.0021146, -0.0009643, -0.0011252, 0.0011044
7: -0.0082273, -0.0075397, -0.0081912, -0.0075185, -0.0007006, 0.0006515
8: 0.0055076, 0.0094875, 0.0057583, 0.0095828, -0.0040156, 0.0036693
9: -0.0036825, -0.0032020, -0.0036806, -0.0031716, -0.0005109, 0.0004786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006043, upper bound: 0.0005979
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006043, upper bound: 0.0005979
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007886, 0.0011153, 0.0007736, 0.0011157, -0.0003271, 0.0003418
1: 0.9934306, 0.9942092, 0.9934255, 0.9942411, -0.0008106, 0.0007837
2: -0.0085520, -0.0055082, -0.0086039, -0.0056800, -0.0026473, 0.0028706
3: 0.0036746, 0.0041459, 0.0036559, 0.0041493, -0.0004746, 0.0004900
4: 0.0027704, 0.0051761, 0.0029062, 0.0052171, -0.0024467, 0.0022699
5: 0.0052310, 0.0064057, 0.0051954, 0.0063686, -0.0011376, 0.0012103
6: -0.0020896, -0.0010101, -0.0021076, -0.0009771, -0.0011125, 0.0010975
7: -0.0082300, -0.0075397, -0.0081975, -0.0075244, -0.0007001, 0.0006578
8: 0.0054887, 0.0094879, 0.0057145, 0.0095562, -0.0040071, 0.0037127
9: -0.0036827, -0.0032019, -0.0036809, -0.0031801, -0.0005026, 0.0004790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006355, upper bound: 0.0006229
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006355, upper bound: 0.0006229
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0007812, 0.0011155, 0.0007658, 0.0011159, -0.0003347, 0.0003497
1: 0.9934281, 0.9942250, 0.9934230, 0.9942575, -0.0008293, 0.0008019
2: -0.0085776, -0.0055030, -0.0086308, -0.0057112, -0.0026477, 0.0029124
3: 0.0036654, 0.0041475, 0.0036461, 0.0041511, -0.0004856, 0.0005014
4: 0.0027663, 0.0051963, 0.0029308, 0.0052383, -0.0024720, 0.0022654
5: 0.0052135, 0.0064069, 0.0051770, 0.0063619, -0.0011484, 0.0012299
6: -0.0020984, -0.0009939, -0.0021169, -0.0009601, -0.0011383, 0.0011230
7: -0.0082310, -0.0075322, -0.0081916, -0.0075165, -0.0007106, 0.0006594
8: 0.0054820, 0.0095215, 0.0057554, 0.0095914, -0.0040508, 0.0037070
9: -0.0036827, -0.0031912, -0.0036806, -0.0031689, -0.0005139, 0.0004894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006086, upper bound: 0.0006032
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006086, upper bound: 0.0006032
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0007811, 0.0011155, 0.0007718, 0.0011158, -0.0003347, 0.0003438
1: 0.9934281, 0.9942251, 0.9934250, 0.9942449, -0.0008168, 0.0008001
2: -0.0085779, -0.0054891, -0.0086101, -0.0056780, -0.0026801, 0.0029030
3: 0.0036653, 0.0041476, 0.0036536, 0.0041497, -0.0004844, 0.0004940
4: 0.0027553, 0.0051965, 0.0029046, 0.0052220, -0.0024667, 0.0022919
5: 0.0052133, 0.0064099, 0.0051912, 0.0063690, -0.0011558, 0.0012187
6: -0.0020986, -0.0009936, -0.0021098, -0.0009732, -0.0011254, 0.0011161
7: -0.0082336, -0.0075320, -0.0081979, -0.0075226, -0.0007094, 0.0006658
8: 0.0054636, 0.0095219, 0.0057118, 0.0095643, -0.0040413, 0.0037503
9: -0.0036829, -0.0031910, -0.0036809, -0.0031775, -0.0005053, 0.0004899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006428, upper bound: 0.0006322
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006428, upper bound: 0.0006322
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007968, 0.0011151, 0.0008013, 0.0011150, -0.0003182, 0.0003138
1: 0.9934333, 0.9941917, 0.9934347, 0.9941822, -0.0007489, 0.0007570
2: -0.0085235, -0.0055075, -0.0085080, -0.0055472, -0.0027512, 0.0027908
3: 0.0036850, 0.0041440, 0.0036906, 0.0041429, -0.0004580, 0.0004534
4: 0.0027698, 0.0051536, 0.0028013, 0.0051413, -0.0023715, 0.0023523
5: 0.0052506, 0.0064059, 0.0052612, 0.0063973, -0.0011467, 0.0011447
6: -0.0020797, -0.0010281, -0.0020743, -0.0010379, -0.0010417, 0.0010462
7: -0.0082301, -0.0075480, -0.0082226, -0.0075526, -0.0006776, 0.0006746
8: 0.0054877, 0.0094505, 0.0055400, 0.0094302, -0.0038862, 0.0038500
9: -0.0036827, -0.0032138, -0.0036823, -0.0032203, -0.0004624, 0.0004685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005646, upper bound: 0.0005627
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006077, upper bound: 0.0006146
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007968, 0.0011151, 0.0007947, 0.0011152, -0.0003184, 0.0003204
1: 0.9934333, 0.9941917, 0.9934325, 0.9941963, -0.0007630, 0.0007592
2: -0.0085235, -0.0055075, -0.0085309, -0.0055494, -0.0027450, 0.0028125
3: 0.0036850, 0.0041440, 0.0036823, 0.0041445, -0.0004595, 0.0004617
4: 0.0027698, 0.0051536, 0.0028030, 0.0051594, -0.0023896, 0.0023506
5: 0.0052506, 0.0064059, 0.0052455, 0.0063968, -0.0011463, 0.0011604
6: -0.0020797, -0.0010281, -0.0020822, -0.0010235, -0.0010562, 0.0010541
7: -0.0082301, -0.0075480, -0.0082222, -0.0075459, -0.0006843, 0.0006742
8: 0.0054877, 0.0094505, 0.0055428, 0.0094602, -0.0039160, 0.0038464
9: -0.0036827, -0.0032138, -0.0036823, -0.0032107, -0.0004720, 0.0004684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005646, upper bound: 0.0005627
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006077, upper bound: 0.0006146
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007906, 0.0011153, 0.0008060, 0.0011149, -0.0003243, 0.0003093
1: 0.9934312, 0.9942049, 0.9934362, 0.9941725, -0.0007414, 0.0007687
2: -0.0085451, -0.0054894, -0.0084920, -0.0054663, -0.0028444, 0.0027708
3: 0.0036772, 0.0041454, 0.0036964, 0.0041419, -0.0004647, 0.0004490
4: 0.0027555, 0.0051706, 0.0027373, 0.0051286, -0.0023731, 0.0024333
5: 0.0052358, 0.0064098, 0.0052722, 0.0064148, -0.0011790, 0.0011376
6: -0.0020872, -0.0010145, -0.0020687, -0.0010185, -0.0010687, 0.0010543
7: -0.0082335, -0.0075417, -0.0082379, -0.0075573, -0.0006763, 0.0006962
8: 0.0054640, 0.0094788, 0.0054336, 0.0094091, -0.0038832, 0.0039823
9: -0.0036829, -0.0032048, -0.0036831, -0.0032270, -0.0004558, 0.0004783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006602
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006658
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007906, 0.0011153, 0.0008346, 0.0011141, -0.0003235, 0.0002806
1: 0.9934312, 0.9942049, 0.9934456, 0.9941117, -0.0006806, 0.0007593
2: -0.0085451, -0.0054894, -0.0083930, -0.0053349, -0.0029762, 0.0026717
3: 0.0036772, 0.0041454, 0.0037323, 0.0041353, -0.0004582, 0.0004131
4: 0.0027555, 0.0051706, 0.0026334, 0.0050504, -0.0022949, 0.0025372
5: 0.0052358, 0.0064098, 0.0053401, 0.0064432, -0.0012074, 0.0010697
6: -0.0020872, -0.0010145, -0.0020344, -0.0009729, -0.0011143, 0.0010199
7: -0.0082335, -0.0075417, -0.0082628, -0.0075863, -0.0006472, 0.0007211
8: 0.0054640, 0.0094788, 0.0052610, 0.0092790, -0.0037531, 0.0041551
9: -0.0036829, -0.0032048, -0.0036845, -0.0032685, -0.0004143, 0.0004797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006602
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006658
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0008194, 0.0011145, 0.0008054, 0.0011149, -0.0002955, 0.0003092
1: 0.9934405, 0.9941441, 0.9934360, 0.9941737, -0.0007332, 0.0007080
2: -0.0084456, -0.0053585, -0.0084941, -0.0055135, -0.0027139, 0.0029043
3: 0.0037132, 0.0041388, 0.0036957, 0.0041420, -0.0004288, 0.0004432
4: 0.0026521, 0.0050920, 0.0027746, 0.0051303, -0.0024782, 0.0023174
5: 0.0053040, 0.0064381, 0.0052708, 0.0064046, -0.0011006, 0.0011673
6: -0.0020526, -0.0009811, -0.0020695, -0.0010349, -0.0010178, 0.0010884
7: -0.0082583, -0.0075709, -0.0082290, -0.0075567, -0.0007016, 0.0006581
8: 0.0052920, 0.0093482, 0.0054957, 0.0094118, -0.0040571, 0.0037940
9: -0.0036842, -0.0032465, -0.0036826, -0.0032262, -0.0004580, 0.0004362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006387, upper bound: 0.0006339
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006471
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0008194, 0.0011145, 0.0007968, 0.0011151, -0.0002957, 0.0003177
1: 0.9934405, 0.9941441, 0.9934333, 0.9941917, -0.0007512, 0.0007108
2: -0.0084456, -0.0053585, -0.0085235, -0.0055075, -0.0027211, 0.0029349
3: 0.0037132, 0.0041388, 0.0036850, 0.0041440, -0.0004308, 0.0004538
4: 0.0026521, 0.0050920, 0.0027698, 0.0051536, -0.0025014, 0.0023222
5: 0.0053040, 0.0064381, 0.0052506, 0.0064059, -0.0011019, 0.0011875
6: -0.0020526, -0.0009811, -0.0020797, -0.0010281, -0.0010245, 0.0010986
7: -0.0082583, -0.0075709, -0.0082301, -0.0075480, -0.0007103, 0.0006593
8: 0.0052920, 0.0093482, 0.0054877, 0.0094505, -0.0040959, 0.0038024
9: -0.0036842, -0.0032465, -0.0036827, -0.0032138, -0.0004704, 0.0004362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006387, upper bound: 0.0006339
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006472
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007886, 0.0011153, 0.0007985, 0.0011151, -0.0003265, 0.0003168
1: 0.9934305, 0.9942093, 0.9934337, 0.9941882, -0.0007578, 0.0007756
2: -0.0085521, -0.0054873, -0.0085178, -0.0054486, -0.0028742, 0.0027975
3: 0.0036747, 0.0041459, 0.0036870, 0.0041436, -0.0004689, 0.0004588
4: 0.0027539, 0.0051761, 0.0027233, 0.0051490, -0.0023951, 0.0024528
5: 0.0052310, 0.0064103, 0.0052545, 0.0064186, -0.0011876, 0.0011558
6: -0.0020896, -0.0010100, -0.0020777, -0.0010123, -0.0010773, 0.0010677
7: -0.0082339, -0.0075396, -0.0082413, -0.0075497, -0.0006843, 0.0007016
8: 0.0054613, 0.0094880, 0.0054104, 0.0094430, -0.0039194, 0.0040159
9: -0.0036829, -0.0032018, -0.0036833, -0.0032162, -0.0004667, 0.0004814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006647
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006672
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007886, 0.0011153, 0.0008279, 0.0011143, -0.0003257, 0.0002874
1: 0.9934305, 0.9942093, 0.9934434, 0.9941258, -0.0006953, 0.0007659
2: -0.0085521, -0.0054873, -0.0084161, -0.0053170, -0.0030089, 0.0026968
3: 0.0036747, 0.0041459, 0.0037239, 0.0041369, -0.0004622, 0.0004219
4: 0.0027539, 0.0051761, 0.0026193, 0.0050686, -0.0023147, 0.0025568
5: 0.0052310, 0.0064103, 0.0053243, 0.0064471, -0.0012161, 0.0010860
6: -0.0020896, -0.0010100, -0.0020424, -0.0009667, -0.0011229, 0.0010324
7: -0.0082339, -0.0075396, -0.0082661, -0.0075795, -0.0006544, 0.0007265
8: 0.0054613, 0.0094880, 0.0052375, 0.0093093, -0.0037860, 0.0041889
9: -0.0036829, -0.0032018, -0.0036846, -0.0032588, -0.0004240, 0.0004828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006647
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006672
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0008173, 0.0011146, 0.0007982, 0.0011151, -0.0002978, 0.0003163
1: 0.9934399, 0.9941484, 0.9934337, 0.9941887, -0.0007489, 0.0007147
2: -0.0084528, -0.0053562, -0.0085186, -0.0054952, -0.0027427, 0.0029294
3: 0.0037106, 0.0041393, 0.0036867, 0.0041436, -0.0004331, 0.0004526
4: 0.0026503, 0.0050977, 0.0027601, 0.0051497, -0.0024994, 0.0023375
5: 0.0052991, 0.0064386, 0.0052539, 0.0064085, -0.0011095, 0.0011846
6: -0.0020551, -0.0009803, -0.0020780, -0.0010285, -0.0010266, 0.0010977
7: -0.0082587, -0.0075688, -0.0082324, -0.0075494, -0.0007093, 0.0006637
8: 0.0052890, 0.0093576, 0.0054717, 0.0094441, -0.0040924, 0.0038284
9: -0.0036842, -0.0032434, -0.0036828, -0.0032159, -0.0004684, 0.0004394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006450
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006477
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0008173, 0.0011146, 0.0007893, 0.0011153, -0.0002980, 0.0003253
1: 0.9934399, 0.9941484, 0.9934308, 0.9942078, -0.0007679, 0.0007176
2: -0.0084528, -0.0053562, -0.0085496, -0.0054884, -0.0027526, 0.0029626
3: 0.0037106, 0.0041393, 0.0036755, 0.0041457, -0.0004351, 0.0004637
4: 0.0026503, 0.0050977, 0.0027548, 0.0051742, -0.0025239, 0.0023429
5: 0.0052991, 0.0064386, 0.0052327, 0.0064100, -0.0011109, 0.0012059
6: -0.0020551, -0.0009803, -0.0020887, -0.0010116, -0.0010436, 0.0011085
7: -0.0082587, -0.0075688, -0.0082337, -0.0075404, -0.0007184, 0.0006650
8: 0.0052890, 0.0093576, 0.0054627, 0.0094848, -0.0041337, 0.0038373
9: -0.0036842, -0.0032434, -0.0036829, -0.0032029, -0.0004814, 0.0004395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006450
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006450
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007824, 0.0011155, 0.0008060, 0.0011149, -0.0003325, 0.0003095
1: 0.9934285, 0.9942223, 0.9934362, 0.9941724, -0.0007439, 0.0007861
2: -0.0085734, -0.0054826, -0.0084918, -0.0054702, -0.0028721, 0.0027785
3: 0.0036669, 0.0041473, 0.0036965, 0.0041419, -0.0004750, 0.0004508
4: 0.0027502, 0.0051930, 0.0027404, 0.0051285, -0.0023783, 0.0024526
5: 0.0052163, 0.0064113, 0.0052723, 0.0064140, -0.0011976, 0.0011389
6: -0.0020970, -0.0009965, -0.0020687, -0.0010199, -0.0010771, 0.0010722
7: -0.0082348, -0.0075334, -0.0082372, -0.0075573, -0.0006775, 0.0007038
8: 0.0054551, 0.0095161, 0.0054388, 0.0094089, -0.0038917, 0.0040148
9: -0.0036829, -0.0031929, -0.0036831, -0.0032271, -0.0004558, 0.0004902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006600
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006650
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007824, 0.0011155, 0.0008347, 0.0011141, -0.0003318, 0.0002808
1: 0.9934285, 0.9942223, 0.9934456, 0.9941117, -0.0006832, 0.0007768
2: -0.0085734, -0.0054826, -0.0083928, -0.0053382, -0.0030047, 0.0026794
3: 0.0036669, 0.0041473, 0.0037323, 0.0041353, -0.0004684, 0.0004150
4: 0.0027502, 0.0051930, 0.0026361, 0.0050503, -0.0023001, 0.0025569
5: 0.0052163, 0.0064113, 0.0053402, 0.0064425, -0.0012261, 0.0010710
6: -0.0020970, -0.0009965, -0.0020343, -0.0009740, -0.0011230, 0.0010378
7: -0.0082348, -0.0075334, -0.0082621, -0.0075864, -0.0006484, 0.0007287
8: 0.0054551, 0.0095161, 0.0052654, 0.0092788, -0.0037616, 0.0041882
9: -0.0036829, -0.0031929, -0.0036844, -0.0032686, -0.0004144, 0.0004915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006600
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006650
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0008106, 0.0011148, 0.0007970, 0.0011151, -0.0003045, 0.0003178
1: 0.9934377, 0.9941627, 0.9934332, 0.9941915, -0.0007538, 0.0007295
2: -0.0084761, -0.0053615, -0.0085229, -0.0055292, -0.0027331, 0.0029331
3: 0.0037022, 0.0041408, 0.0036852, 0.0041439, -0.0004418, 0.0004556
4: 0.0026545, 0.0051161, 0.0027870, 0.0051531, -0.0024986, 0.0023291
5: 0.0052831, 0.0064374, 0.0052510, 0.0064012, -0.0011181, 0.0011865
6: -0.0020632, -0.0009821, -0.0020795, -0.0010285, -0.0010347, 0.0010974
7: -0.0082577, -0.0075619, -0.0082260, -0.0075482, -0.0007095, 0.0006641
8: 0.0052960, 0.0093882, 0.0055162, 0.0094497, -0.0040928, 0.0038141
9: -0.0036842, -0.0032337, -0.0036825, -0.0032141, -0.0004701, 0.0004488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006333, upper bound: 0.0006389
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006333, upper bound: 0.0006397
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0008105, 0.0011148, 0.0008058, 0.0011149, -0.0003044, 0.0003089
1: 0.9934376, 0.9941630, 0.9934361, 0.9941728, -0.0007351, 0.0007269
2: -0.0084764, -0.0053493, -0.0084924, -0.0055083, -0.0027565, 0.0029139
3: 0.0037021, 0.0041409, 0.0036962, 0.0041419, -0.0004399, 0.0004446
4: 0.0026448, 0.0051163, 0.0027705, 0.0051290, -0.0024841, 0.0023459
5: 0.0052829, 0.0064401, 0.0052719, 0.0064057, -0.0011229, 0.0011682
6: -0.0020633, -0.0009779, -0.0020689, -0.0010331, -0.0010303, 0.0010910
7: -0.0082600, -0.0075618, -0.0082300, -0.0075572, -0.0007029, 0.0006681
8: 0.0052800, 0.0093886, 0.0054888, 0.0094096, -0.0040685, 0.0038422
9: -0.0036843, -0.0032335, -0.0036827, -0.0032268, -0.0004575, 0.0004491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006407
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006432
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0007800, 0.0011156, 0.0007985, 0.0011151, -0.0003351, 0.0003170
1: 0.9934277, 0.9942275, 0.9934337, 0.9941882, -0.0007605, 0.0007938
2: -0.0085818, -0.0054805, -0.0085177, -0.0054520, -0.0029016, 0.0028053
3: 0.0036639, 0.0041478, 0.0036871, 0.0041436, -0.0004797, 0.0004607
4: 0.0027485, 0.0051996, 0.0027260, 0.0051489, -0.0024004, 0.0024736
5: 0.0052106, 0.0064117, 0.0052546, 0.0064179, -0.0012073, 0.0011571
6: -0.0020999, -0.0009912, -0.0020776, -0.0010135, -0.0010864, 0.0010865
7: -0.0082352, -0.0075309, -0.0082406, -0.0075497, -0.0006855, 0.0007097
8: 0.0054524, 0.0095270, 0.0054149, 0.0094428, -0.0039281, 0.0040510
9: -0.0036830, -0.0031894, -0.0036833, -0.0032163, -0.0004667, 0.0004938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006637
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006657
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0007800, 0.0011156, 0.0008280, 0.0011143, -0.0003343, 0.0002876
1: 0.9934277, 0.9942275, 0.9934434, 0.9941257, -0.0006980, 0.0007842
2: -0.0085818, -0.0054805, -0.0084159, -0.0053193, -0.0030378, 0.0027046
3: 0.0036639, 0.0041478, 0.0037240, 0.0041368, -0.0004729, 0.0004239
4: 0.0027485, 0.0051996, 0.0026211, 0.0050685, -0.0023200, 0.0025785
5: 0.0052106, 0.0064117, 0.0053244, 0.0064466, -0.0012359, 0.0010873
6: -0.0020999, -0.0009912, -0.0020423, -0.0009675, -0.0011324, 0.0010511
7: -0.0082352, -0.0075309, -0.0082657, -0.0075796, -0.0006556, 0.0007348
8: 0.0054524, 0.0095270, 0.0052405, 0.0093091, -0.0037947, 0.0042263
9: -0.0036830, -0.0031894, -0.0036846, -0.0032589, -0.0004241, 0.0004952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006637
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006657
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0008085, 0.0011148, 0.0007901, 0.0011153, -0.0003068, 0.0003247
1: 0.9934370, 0.9941671, 0.9934310, 0.9942060, -0.0007690, 0.0007361
2: -0.0084834, -0.0053596, -0.0085467, -0.0055096, -0.0027663, 0.0029578
3: 0.0036995, 0.0041413, 0.0036766, 0.0041455, -0.0004460, 0.0004647
4: 0.0026529, 0.0051218, 0.0027715, 0.0051719, -0.0025190, 0.0023503
5: 0.0052782, 0.0064379, 0.0052346, 0.0064054, -0.0011273, 0.0012032
6: -0.0020657, -0.0009814, -0.0020877, -0.0010134, -0.0010523, 0.0011063
7: -0.0082581, -0.0075598, -0.0082297, -0.0075412, -0.0007169, 0.0006699
8: 0.0052934, 0.0093977, 0.0054906, 0.0094810, -0.0041263, 0.0038507
9: -0.0036842, -0.0032306, -0.0036827, -0.0032041, -0.0004801, 0.0004520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006319, upper bound: 0.0006387
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006319, upper bound: 0.0006397
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0008084, 0.0011148, 0.0007987, 0.0011151, -0.0003067, 0.0003161
1: 0.9934370, 0.9941674, 0.9934338, 0.9941878, -0.0007508, 0.0007336
2: -0.0084837, -0.0053473, -0.0085170, -0.0054900, -0.0027873, 0.0029394
3: 0.0036994, 0.0041413, 0.0036874, 0.0041435, -0.0004441, 0.0004540
4: 0.0026432, 0.0051220, 0.0027560, 0.0051484, -0.0025052, 0.0023661
5: 0.0052780, 0.0064405, 0.0052551, 0.0064097, -0.0011317, 0.0011855
6: -0.0020658, -0.0009772, -0.0020774, -0.0010267, -0.0010391, 0.0011002
7: -0.0082604, -0.0075597, -0.0082334, -0.0075499, -0.0007105, 0.0006737
8: 0.0052773, 0.0093981, 0.0054648, 0.0094419, -0.0041031, 0.0038773
9: -0.0036843, -0.0032305, -0.0036829, -0.0032165, -0.0004678, 0.0004523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006407
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006432
time: 0.71 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.26 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006230, upper bound: 0.0006366
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006419, upper bound: 0.0006630
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006230, upper bound: 0.0006366
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006419, upper bound: 0.0006630
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005982, upper bound: 0.0006037
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006256, upper bound: 0.0006354
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005982, upper bound: 0.0006037
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006256, upper bound: 0.0006354
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006245, upper bound: 0.0006374
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006465, upper bound: 0.0006668
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006245, upper bound: 0.0006374
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006465, upper bound: 0.0006668
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006034, upper bound: 0.0006080
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006351, upper bound: 0.0006429
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006034, upper bound: 0.0006080
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006351, upper bound: 0.0006428
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006054
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006241
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006054
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006241
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005534, upper bound: 0.0005491
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005534, upper bound: 0.0005491
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005534, upper bound: 0.0005491
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005534, upper bound: 0.0005491
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006054
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006367
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006054
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006367
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005534, upper bound: 0.0005491
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005534, upper bound: 0.0005491
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005534, upper bound: 0.0005491
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005534, upper bound: 0.0006083
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005836, upper bound: 0.0005852
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005836, upper bound: 0.0005852
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005836, upper bound: 0.0005852
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005836, upper bound: 0.0006043
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006120, upper bound: 0.0006131
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006120, upper bound: 0.0006355
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006120, upper bound: 0.0006131
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006120, upper bound: 0.0006355
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005831, upper bound: 0.0005829
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005831, upper bound: 0.0006086
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005831, upper bound: 0.0005829
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005831, upper bound: 0.0006086
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006120, upper bound: 0.0006123
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006120, upper bound: 0.0006428
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006120, upper bound: 0.0006122
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006120, upper bound: 0.0006428
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006083
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006083
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006354, upper bound: 0.0006256
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006354, upper bound: 0.0006256
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006354, upper bound: 0.0006256
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006354, upper bound: 0.0006256
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006429, upper bound: 0.0006351
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006429, upper bound: 0.0006351
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006429, upper bound: 0.0006351
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006429, upper bound: 0.0006351
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005543, upper bound: 0.0005541
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006078, upper bound: 0.0006160
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005543, upper bound: 0.0005541
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006078, upper bound: 0.0006160
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005660, upper bound: 0.0005655
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006295
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005660, upper bound: 0.0005655
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006295
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006043, upper bound: 0.0005979
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006043, upper bound: 0.0005979
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006355, upper bound: 0.0006229
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006355, upper bound: 0.0006229
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006086, upper bound: 0.0006032
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006086, upper bound: 0.0006032
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006428, upper bound: 0.0006322
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006428, upper bound: 0.0006322
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005646, upper bound: 0.0005627
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006077, upper bound: 0.0006146
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0005646, upper bound: 0.0005627
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006077, upper bound: 0.0006146
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006602
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006658
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006602
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006658
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006387, upper bound: 0.0006339
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006471
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006387, upper bound: 0.0006339
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006472
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006647
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006672
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006647
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006672
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006450
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006477
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006450
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006450
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006600
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006650
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006600
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006650
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006333, upper bound: 0.0006389
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006333, upper bound: 0.0006397
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006407
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006432
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006637
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006657
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006637
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006657
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006319, upper bound: 0.0006387
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006319, upper bound: 0.0006397
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006407
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.26
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006432

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007678, 0.0011159, 0.0008067, 0.0011149, -0.0003471, 0.0003092
1: 0.9934236, 0.9942534, 0.9934365, 0.9941710, -0.0007474, 0.0008169
2: -0.0086240, -0.0057492, -0.0084896, -0.0055607, -0.0028409, 0.0025129
3: 0.0036486, 0.0041506, 0.0036973, 0.0041417, -0.0004931, 0.0004534
4: 0.0029608, 0.0052329, 0.0028119, 0.0051268, -0.0021659, 0.0024211
5: 0.0051816, 0.0063537, 0.0052738, 0.0063944, -0.0012128, 0.0010799
6: -0.0021146, -0.0009644, -0.0020679, -0.0010496, -0.0010649, 0.0011035
7: -0.0081844, -0.0075185, -0.0082201, -0.0075580, -0.0006265, 0.0006920
8: 0.0058053, 0.0095825, 0.0055577, 0.0094060, -0.0035396, 0.0039652
9: -0.0036802, -0.0031717, -0.0036821, -0.0032280, -0.0004522, 0.0005104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006655, upper bound: 0.0006678
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006655, upper bound: 0.0006678
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007736, 0.0011157, 0.0008065, 0.0011149, -0.0003412, 0.0003092
1: 0.9934256, 0.9942409, 0.9934363, 0.9941711, -0.0007455, 0.0008046
2: -0.0086038, -0.0057135, -0.0084900, -0.0055449, -0.0028334, 0.0025484
3: 0.0036559, 0.0041493, 0.0036971, 0.0041417, -0.0004858, 0.0004522
4: 0.0029327, 0.0052170, 0.0027994, 0.0051270, -0.0021944, 0.0024175
5: 0.0051956, 0.0063614, 0.0052736, 0.0063978, -0.0012022, 0.0010878
6: -0.0021075, -0.0009772, -0.0020680, -0.0010458, -0.0010617, 0.0010908
7: -0.0081912, -0.0075245, -0.0082230, -0.0075579, -0.0006333, 0.0006918
8: 0.0057585, 0.0095559, 0.0055370, 0.0094064, -0.0035869, 0.0039582
9: -0.0036806, -0.0031802, -0.0036823, -0.0032279, -0.0004527, 0.0005021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006688, upper bound: 0.0006787
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006688, upper bound: 0.0006787
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007678, 0.0011159, 0.0008353, 0.0011141, -0.0003463, 0.0002806
1: 0.9934236, 0.9942534, 0.9934458, 0.9941103, -0.0006867, 0.0008076
2: -0.0086240, -0.0057492, -0.0083907, -0.0054275, -0.0029733, 0.0024140
3: 0.0036486, 0.0041506, 0.0037331, 0.0041352, -0.0004866, 0.0004175
4: 0.0029608, 0.0052329, 0.0027066, 0.0050486, -0.0020877, 0.0025263
5: 0.0051816, 0.0063537, 0.0053417, 0.0064232, -0.0012415, 0.0010120
6: -0.0021146, -0.0009644, -0.0020336, -0.0010050, -0.0011095, 0.0010692
7: -0.0081844, -0.0075185, -0.0082452, -0.0075870, -0.0005974, 0.0007144
8: 0.0058053, 0.0095825, 0.0053827, 0.0092760, -0.0034096, 0.0041399
9: -0.0036802, -0.0031717, -0.0036835, -0.0032695, -0.0004107, 0.0005118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005903, upper bound: 0.0006004
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005903, upper bound: 0.0006004
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007736, 0.0011157, 0.0008352, 0.0011141, -0.0003405, 0.0002805
1: 0.9934256, 0.9942409, 0.9934458, 0.9941105, -0.0006849, 0.0007951
2: -0.0086038, -0.0057135, -0.0083910, -0.0054130, -0.0029646, 0.0024496
3: 0.0036559, 0.0041493, 0.0037330, 0.0041352, -0.0004793, 0.0004163
4: 0.0029327, 0.0052170, 0.0026952, 0.0050488, -0.0021161, 0.0025218
5: 0.0051956, 0.0063614, 0.0053415, 0.0064263, -0.0012308, 0.0010199
6: -0.0021075, -0.0009772, -0.0020337, -0.0010000, -0.0011075, 0.0010564
7: -0.0081912, -0.0075245, -0.0082480, -0.0075869, -0.0006043, 0.0007137
8: 0.0057585, 0.0095559, 0.0053636, 0.0092764, -0.0034569, 0.0041318
9: -0.0036806, -0.0031802, -0.0036837, -0.0032694, -0.0004112, 0.0005035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006392
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006630
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007598, 0.0011161, 0.0008067, 0.0011149, -0.0003551, 0.0003094
1: 0.9934211, 0.9942704, 0.9934365, 0.9941710, -0.0007499, 0.0008340
2: -0.0086516, -0.0057519, -0.0084896, -0.0055607, -0.0028694, 0.0025148
3: 0.0036386, 0.0041524, 0.0036973, 0.0041417, -0.0005031, 0.0004552
4: 0.0029630, 0.0052548, 0.0028119, 0.0051268, -0.0021637, 0.0024429
5: 0.0051627, 0.0063531, 0.0052738, 0.0063944, -0.0012317, 0.0010793
6: -0.0021241, -0.0009469, -0.0020679, -0.0010496, -0.0010745, 0.0011210
7: -0.0081839, -0.0075104, -0.0082201, -0.0075580, -0.0006259, 0.0006980
8: 0.0058089, 0.0096188, 0.0055577, 0.0094060, -0.0035373, 0.0040019
9: -0.0036802, -0.0031602, -0.0036821, -0.0032280, -0.0004522, 0.0005220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006567, upper bound: 0.0006479
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006567, upper bound: 0.0006479
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007675, 0.0011159, 0.0008065, 0.0011149, -0.0003473, 0.0003093
1: 0.9934236, 0.9942539, 0.9934363, 0.9941711, -0.0007475, 0.0008175
2: -0.0086247, -0.0057215, -0.0084900, -0.0055449, -0.0028584, 0.0025432
3: 0.0036483, 0.0041507, 0.0036971, 0.0041417, -0.0004934, 0.0004535
4: 0.0029390, 0.0052335, 0.0027994, 0.0051270, -0.0021880, 0.0024341
5: 0.0051811, 0.0063596, 0.0052736, 0.0063978, -0.0012167, 0.0010860
6: -0.0021148, -0.0009639, -0.0020680, -0.0010458, -0.0010690, 0.0011041
7: -0.0081897, -0.0075183, -0.0082230, -0.0075579, -0.0006318, 0.0006975
8: 0.0057690, 0.0095835, 0.0055370, 0.0094064, -0.0035773, 0.0039865
9: -0.0036805, -0.0031714, -0.0036823, -0.0032279, -0.0004526, 0.0005109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006609, upper bound: 0.0006585
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006609, upper bound: 0.0006585
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007598, 0.0011161, 0.0008353, 0.0011141, -0.0003543, 0.0002808
1: 0.9934211, 0.9942704, 0.9934458, 0.9941103, -0.0006892, 0.0008246
2: -0.0086516, -0.0057519, -0.0083907, -0.0054275, -0.0030018, 0.0024159
3: 0.0036386, 0.0041524, 0.0037331, 0.0041352, -0.0004966, 0.0004193
4: 0.0029630, 0.0052548, 0.0027066, 0.0050486, -0.0020855, 0.0025481
5: 0.0051627, 0.0063531, 0.0053417, 0.0064232, -0.0012604, 0.0010113
6: -0.0021241, -0.0009469, -0.0020336, -0.0010050, -0.0011191, 0.0010867
7: -0.0081839, -0.0075104, -0.0082452, -0.0075870, -0.0005969, 0.0007203
8: 0.0058089, 0.0096188, 0.0053827, 0.0092760, -0.0034073, 0.0041766
9: -0.0036802, -0.0031602, -0.0036835, -0.0032695, -0.0004107, 0.0005234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005930, upper bound: 0.0005990
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005914, upper bound: 0.0005979
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007675, 0.0011159, 0.0008352, 0.0011141, -0.0003466, 0.0002807
1: 0.9934236, 0.9942539, 0.9934458, 0.9941105, -0.0006869, 0.0008081
2: -0.0086247, -0.0057215, -0.0083910, -0.0054130, -0.0029895, 0.0024444
3: 0.0036483, 0.0041507, 0.0037330, 0.0041352, -0.0004869, 0.0004177
4: 0.0029390, 0.0052335, 0.0026952, 0.0050488, -0.0021098, 0.0025384
5: 0.0051811, 0.0063596, 0.0053415, 0.0064263, -0.0012452, 0.0010181
6: -0.0021148, -0.0009639, -0.0020337, -0.0010000, -0.0011148, 0.0010697
7: -0.0081897, -0.0075183, -0.0082480, -0.0075869, -0.0006027, 0.0007194
8: 0.0057690, 0.0095835, 0.0053636, 0.0092764, -0.0034473, 0.0041601
9: -0.0036805, -0.0031714, -0.0036837, -0.0032694, -0.0004111, 0.0005122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006026, upper bound: 0.0006092
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006026, upper bound: 0.0006354
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007658, 0.0011159, 0.0007992, 0.0011151, -0.0003492, 0.0003167
1: 0.9934230, 0.9942575, 0.9934340, 0.9941869, -0.0007639, 0.0008234
2: -0.0086306, -0.0057470, -0.0085154, -0.0055415, -0.0028666, 0.0025420
3: 0.0036462, 0.0041511, 0.0036879, 0.0041434, -0.0004972, 0.0004631
4: 0.0029591, 0.0052381, 0.0027967, 0.0051471, -0.0021880, 0.0024414
5: 0.0051771, 0.0063541, 0.0052562, 0.0063985, -0.0012214, 0.0010980
6: -0.0021168, -0.0009602, -0.0020769, -0.0010333, -0.0010835, 0.0011166
7: -0.0081848, -0.0075166, -0.0082237, -0.0075504, -0.0006344, 0.0007007
8: 0.0058025, 0.0095911, 0.0055325, 0.0094398, -0.0035764, 0.0039980
9: -0.0036802, -0.0031690, -0.0036823, -0.0032172, -0.0004630, 0.0005134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006781, upper bound: 0.0006718
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006781, upper bound: 0.0006718
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007718, 0.0011158, 0.0007991, 0.0011151, -0.0003432, 0.0003167
1: 0.9934250, 0.9942448, 0.9934340, 0.9941869, -0.0007619, 0.0008108
2: -0.0086100, -0.0057116, -0.0085157, -0.0055253, -0.0028594, 0.0025767
3: 0.0036537, 0.0041497, 0.0036878, 0.0041435, -0.0004898, 0.0004619
4: 0.0029311, 0.0052219, 0.0027839, 0.0051474, -0.0022163, 0.0024380
5: 0.0051913, 0.0063618, 0.0052559, 0.0064021, -0.0012108, 0.0011059
6: -0.0021097, -0.0009733, -0.0020770, -0.0010331, -0.0010766, 0.0011037
7: -0.0081915, -0.0075227, -0.0082268, -0.0075503, -0.0006412, 0.0006997
8: 0.0057559, 0.0095641, 0.0055111, 0.0094402, -0.0036233, 0.0039918
9: -0.0036806, -0.0031776, -0.0036825, -0.0032171, -0.0004635, 0.0005049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006873, upper bound: 0.0006872
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006873, upper bound: 0.0006872
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007658, 0.0011159, 0.0008286, 0.0011143, -0.0003484, 0.0002873
1: 0.9934230, 0.9942575, 0.9934436, 0.9941244, -0.0007014, 0.0008138
2: -0.0086306, -0.0057470, -0.0084138, -0.0054082, -0.0029985, 0.0024427
3: 0.0036462, 0.0041511, 0.0037247, 0.0041367, -0.0004905, 0.0004263
4: 0.0029591, 0.0052381, 0.0026914, 0.0050668, -0.0021077, 0.0025468
5: 0.0051771, 0.0063541, 0.0053259, 0.0064274, -0.0012502, 0.0010283
6: -0.0021168, -0.0009602, -0.0020416, -0.0009983, -0.0011185, 0.0010814
7: -0.0081848, -0.0075166, -0.0082489, -0.0075802, -0.0006046, 0.0007225
8: 0.0058025, 0.0095911, 0.0053573, 0.0093063, -0.0034433, 0.0041735
9: -0.0036802, -0.0031690, -0.0036837, -0.0032598, -0.0004204, 0.0005147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006100, upper bound: 0.0006191
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006100, upper bound: 0.0006374
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007718, 0.0011158, 0.0008285, 0.0011143, -0.0003425, 0.0002873
1: 0.9934250, 0.9942448, 0.9934436, 0.9941245, -0.0006995, 0.0008011
2: -0.0086100, -0.0057116, -0.0084141, -0.0053931, -0.0029908, 0.0024787
3: 0.0036537, 0.0041497, 0.0037246, 0.0041367, -0.0004830, 0.0004251
4: 0.0029311, 0.0052219, 0.0026794, 0.0050671, -0.0021360, 0.0025425
5: 0.0051913, 0.0063618, 0.0053257, 0.0064306, -0.0012393, 0.0010361
6: -0.0021097, -0.0009733, -0.0020417, -0.0009931, -0.0011166, 0.0010684
7: -0.0081915, -0.0075227, -0.0082518, -0.0075801, -0.0006114, 0.0007212
8: 0.0057559, 0.0095641, 0.0053374, 0.0093067, -0.0034904, 0.0041660
9: -0.0036806, -0.0031776, -0.0036839, -0.0032597, -0.0004209, 0.0005063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006367, upper bound: 0.0006599
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006367, upper bound: 0.0006668
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007581, 0.0011161, 0.0007992, 0.0011151, -0.0003569, 0.0003169
1: 0.9934205, 0.9942738, 0.9934340, 0.9941869, -0.0007663, 0.0008398
2: -0.0086573, -0.0057498, -0.0085154, -0.0055415, -0.0028948, 0.0025440
3: 0.0036366, 0.0041528, 0.0036879, 0.0041434, -0.0005069, 0.0004649
4: 0.0029613, 0.0052592, 0.0027967, 0.0051471, -0.0021858, 0.0024625
5: 0.0051588, 0.0063535, 0.0052562, 0.0063985, -0.0012397, 0.0010974
6: -0.0021261, -0.0009433, -0.0020769, -0.0010333, -0.0010928, 0.0011336
7: -0.0081843, -0.0075088, -0.0082237, -0.0075504, -0.0006339, 0.0007073
8: 0.0058061, 0.0096262, 0.0055325, 0.0094398, -0.0035742, 0.0040339
9: -0.0036802, -0.0031578, -0.0036823, -0.0032172, -0.0004630, 0.0005246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006732, upper bound: 0.0006535
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006732, upper bound: 0.0006535
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007658, 0.0011159, 0.0007991, 0.0011151, -0.0003493, 0.0003168
1: 0.9934230, 0.9942576, 0.9934340, 0.9941869, -0.0007639, 0.0008236
2: -0.0086308, -0.0057195, -0.0085157, -0.0055253, -0.0028836, 0.0025716
3: 0.0036462, 0.0041511, 0.0036878, 0.0041435, -0.0004973, 0.0004632
4: 0.0029374, 0.0052383, 0.0027839, 0.0051474, -0.0022100, 0.0024544
5: 0.0051770, 0.0063601, 0.0052559, 0.0064021, -0.0012251, 0.0011042
6: -0.0021169, -0.0009601, -0.0020770, -0.0010331, -0.0010838, 0.0011169
7: -0.0081900, -0.0075165, -0.0082268, -0.0075503, -0.0006397, 0.0007054
8: 0.0057663, 0.0095914, 0.0055111, 0.0094402, -0.0036139, 0.0040201
9: -0.0036805, -0.0031689, -0.0036825, -0.0032171, -0.0004634, 0.0005136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006785, upper bound: 0.0006631
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006785, upper bound: 0.0006631
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007581, 0.0011161, 0.0008286, 0.0011143, -0.0003562, 0.0002875
1: 0.9934205, 0.9942738, 0.9934436, 0.9941244, -0.0007039, 0.0008302
2: -0.0086573, -0.0057498, -0.0084138, -0.0054082, -0.0030267, 0.0024447
3: 0.0036366, 0.0041528, 0.0037247, 0.0041367, -0.0005002, 0.0004281
4: 0.0029613, 0.0052592, 0.0026914, 0.0050668, -0.0021055, 0.0025679
5: 0.0051588, 0.0063535, 0.0053259, 0.0064274, -0.0012685, 0.0010277
6: -0.0021261, -0.0009433, -0.0020416, -0.0009983, -0.0011278, 0.0010983
7: -0.0081843, -0.0075088, -0.0082489, -0.0075802, -0.0006041, 0.0007291
8: 0.0058061, 0.0096262, 0.0053573, 0.0093063, -0.0034410, 0.0042095
9: -0.0036802, -0.0031578, -0.0036837, -0.0032598, -0.0004204, 0.0005259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005803, upper bound: 0.0005796
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005803, upper bound: 0.0006080
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007658, 0.0011159, 0.0008285, 0.0011143, -0.0003485, 0.0002874
1: 0.9934230, 0.9942576, 0.9934436, 0.9941245, -0.0007015, 0.0008140
2: -0.0086308, -0.0057195, -0.0084141, -0.0053931, -0.0030151, 0.0024736
3: 0.0036462, 0.0041511, 0.0037246, 0.0041367, -0.0004906, 0.0004265
4: 0.0029374, 0.0052383, 0.0026794, 0.0050671, -0.0021297, 0.0025589
5: 0.0051770, 0.0063601, 0.0053257, 0.0064306, -0.0012536, 0.0010344
6: -0.0021169, -0.0009601, -0.0020417, -0.0009931, -0.0011238, 0.0010816
7: -0.0081900, -0.0075165, -0.0082518, -0.0075801, -0.0006099, 0.0007269
8: 0.0057663, 0.0095914, 0.0053374, 0.0093067, -0.0034810, 0.0041943
9: -0.0036805, -0.0031689, -0.0036839, -0.0032597, -0.0004208, 0.0005150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006310, upper bound: 0.0006344
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006310, upper bound: 0.0006429
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0008160, 0.0011146, 0.0007885, 0.0011153, -0.0002993, 0.0003261
1: 0.9934395, 0.9941513, 0.9934304, 0.9942093, -0.0007698, 0.0007209
2: -0.0084573, -0.0055695, -0.0085523, -0.0057032, -0.0025417, 0.0027558
3: 0.0037090, 0.0041396, 0.0036746, 0.0041459, -0.0004369, 0.0004650
4: 0.0028189, 0.0051012, 0.0029245, 0.0051763, -0.0023574, 0.0021767
5: 0.0052960, 0.0063925, 0.0052309, 0.0063636, -0.0010676, 0.0011616
6: -0.0020567, -0.0010543, -0.0020897, -0.0010099, -0.0010468, 0.0010353
7: -0.0082184, -0.0075674, -0.0081931, -0.0075396, -0.0006788, 0.0006257
8: 0.0055693, 0.0093635, 0.0057450, 0.0094883, -0.0038583, 0.0035615
9: -0.0036821, -0.0032416, -0.0036807, -0.0032018, -0.0004803, 0.0004391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006054
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006054
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0008107, 0.0011148, 0.0007885, 0.0011153, -0.0003046, 0.0003262
1: 0.9934377, 0.9941624, 0.9934304, 0.9942093, -0.0007716, 0.0007319
2: -0.0084756, -0.0055531, -0.0085523, -0.0057032, -0.0025592, 0.0027709
3: 0.0037024, 0.0041408, 0.0036746, 0.0041459, -0.0004435, 0.0004662
4: 0.0028059, 0.0051156, 0.0029245, 0.0051763, -0.0023703, 0.0021911
5: 0.0052835, 0.0063960, 0.0052309, 0.0063636, -0.0010801, 0.0011652
6: -0.0020630, -0.0010486, -0.0020897, -0.0010099, -0.0010531, 0.0010410
7: -0.0082215, -0.0075621, -0.0081931, -0.0075396, -0.0006819, 0.0006310
8: 0.0055478, 0.0093875, 0.0057450, 0.0094883, -0.0038795, 0.0035856
9: -0.0036822, -0.0032339, -0.0036807, -0.0032018, -0.0004804, 0.0004468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006241
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006241
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0008160, 0.0011146, 0.0008054, 0.0011149, -0.0002989, 0.0003092
1: 0.9934395, 0.9941513, 0.9934360, 0.9941737, -0.0007342, 0.0007153
2: -0.0084573, -0.0055695, -0.0084941, -0.0055135, -0.0027333, 0.0026990
3: 0.0037090, 0.0041396, 0.0036957, 0.0041420, -0.0004330, 0.0004439
4: 0.0028189, 0.0051012, 0.0027746, 0.0051303, -0.0023114, 0.0023266
5: 0.0052960, 0.0063925, 0.0052708, 0.0064046, -0.0011086, 0.0011217
6: -0.0020567, -0.0010543, -0.0020695, -0.0010349, -0.0010218, 0.0010151
7: -0.0082184, -0.0075674, -0.0082290, -0.0075567, -0.0006617, 0.0006616
8: 0.0055693, 0.0093635, 0.0054957, 0.0094118, -0.0037823, 0.0038112
9: -0.0036821, -0.0032416, -0.0036826, -0.0032262, -0.0004559, 0.0004411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005742, upper bound: 0.0005765
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006279, upper bound: 0.0006054
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0008107, 0.0011148, 0.0008054, 0.0011149, -0.0003042, 0.0003094
1: 0.9934377, 0.9941624, 0.9934360, 0.9941737, -0.0007360, 0.0007263
2: -0.0084756, -0.0055531, -0.0084941, -0.0055135, -0.0027509, 0.0027142
3: 0.0037024, 0.0041408, 0.0036957, 0.0041420, -0.0004397, 0.0004451
4: 0.0028059, 0.0051156, 0.0027746, 0.0051303, -0.0023243, 0.0023411
5: 0.0052835, 0.0063960, 0.0052708, 0.0064046, -0.0011211, 0.0011252
6: -0.0020630, -0.0010486, -0.0020695, -0.0010349, -0.0010282, 0.0010208
7: -0.0082215, -0.0075621, -0.0082290, -0.0075567, -0.0006648, 0.0006669
8: 0.0055478, 0.0093875, 0.0054957, 0.0094118, -0.0038035, 0.0038353
9: -0.0036822, -0.0032339, -0.0036826, -0.0032262, -0.0004561, 0.0004487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005742, upper bound: 0.0006048
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006279, upper bound: 0.0006365
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0008160, 0.0011146, 0.0007824, 0.0011155, -0.0002995, 0.0003322
1: 0.9934395, 0.9941513, 0.9934284, 0.9942224, -0.0007829, 0.0007229
2: -0.0084573, -0.0055695, -0.0085735, -0.0056865, -0.0025571, 0.0027760
3: 0.0037090, 0.0041396, 0.0036669, 0.0041473, -0.0004383, 0.0004727
4: 0.0028189, 0.0051012, 0.0029113, 0.0051930, -0.0023741, 0.0021899
5: 0.0052960, 0.0063925, 0.0052163, 0.0063672, -0.0010712, 0.0011762
6: -0.0020567, -0.0010543, -0.0020970, -0.0009965, -0.0010602, 0.0010427
7: -0.0082184, -0.0075674, -0.0081963, -0.0075334, -0.0006850, 0.0006288
8: 0.0055693, 0.0093635, 0.0057229, 0.0095161, -0.0038858, 0.0035832
9: -0.0036821, -0.0032416, -0.0036809, -0.0031929, -0.0004892, 0.0004393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006054
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006054
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0008107, 0.0011148, 0.0007824, 0.0011155, -0.0003048, 0.0003324
1: 0.9934377, 0.9941624, 0.9934284, 0.9942224, -0.0007847, 0.0007340
2: -0.0084756, -0.0055531, -0.0085735, -0.0056865, -0.0025726, 0.0027890
3: 0.0037024, 0.0041408, 0.0036669, 0.0041473, -0.0004449, 0.0004739
4: 0.0028059, 0.0051156, 0.0029113, 0.0051930, -0.0023871, 0.0022044
5: 0.0052835, 0.0063960, 0.0052163, 0.0063672, -0.0010837, 0.0011797
6: -0.0020630, -0.0010486, -0.0020970, -0.0009965, -0.0010666, 0.0010484
7: -0.0082215, -0.0075621, -0.0081963, -0.0075334, -0.0006881, 0.0006342
8: 0.0055478, 0.0093875, 0.0057229, 0.0095161, -0.0039065, 0.0036066
9: -0.0036822, -0.0032339, -0.0036809, -0.0031929, -0.0004893, 0.0004469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006367
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006367
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0008160, 0.0011146, 0.0007982, 0.0011151, -0.0002991, 0.0003164
1: 0.9934395, 0.9941513, 0.9934337, 0.9941887, -0.0007492, 0.0007176
2: -0.0084573, -0.0055695, -0.0085186, -0.0054952, -0.0027558, 0.0027281
3: 0.0037090, 0.0041396, 0.0036867, 0.0041436, -0.0004347, 0.0004528
4: 0.0028189, 0.0051012, 0.0027601, 0.0051497, -0.0023308, 0.0023410
5: 0.0052960, 0.0063925, 0.0052539, 0.0064085, -0.0011125, 0.0011385
6: -0.0020567, -0.0010543, -0.0020780, -0.0010285, -0.0010282, 0.0010237
7: -0.0082184, -0.0075674, -0.0082324, -0.0075494, -0.0006690, 0.0006650
8: 0.0055693, 0.0093635, 0.0054717, 0.0094441, -0.0038157, 0.0038365
9: -0.0036821, -0.0032416, -0.0036828, -0.0032159, -0.0004662, 0.0004412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005634, upper bound: 0.0005635
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006279, upper bound: 0.0006054
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0008107, 0.0011148, 0.0007982, 0.0011151, -0.0003044, 0.0003165
1: 0.9934377, 0.9941624, 0.9934337, 0.9941887, -0.0007510, 0.0007287
2: -0.0084756, -0.0055531, -0.0085186, -0.0054952, -0.0027699, 0.0027400
3: 0.0037024, 0.0041408, 0.0036867, 0.0041436, -0.0004413, 0.0004541
4: 0.0028059, 0.0051156, 0.0027601, 0.0051497, -0.0023438, 0.0023555
5: 0.0052835, 0.0063960, 0.0052539, 0.0064085, -0.0011251, 0.0011421
6: -0.0020630, -0.0010486, -0.0020780, -0.0010285, -0.0010345, 0.0010294
7: -0.0082215, -0.0075621, -0.0082324, -0.0075494, -0.0006720, 0.0006703
8: 0.0055478, 0.0093875, 0.0054717, 0.0094441, -0.0038361, 0.0038596
9: -0.0036822, -0.0032339, -0.0036828, -0.0032159, -0.0004664, 0.0004489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005634, upper bound: 0.0006042
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006279, upper bound: 0.0006367
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0008041, 0.0011149, 0.0007982, 0.0011151, -0.0003109, 0.0003167
1: 0.9934356, 0.9941763, 0.9934337, 0.9941887, -0.0007531, 0.0007426
2: -0.0084983, -0.0055554, -0.0085186, -0.0054952, -0.0027950, 0.0027372
3: 0.0036941, 0.0041423, 0.0036867, 0.0041436, -0.0004495, 0.0004556
4: 0.0028077, 0.0051336, 0.0027601, 0.0051497, -0.0023420, 0.0023735
5: 0.0052679, 0.0063955, 0.0052539, 0.0064085, -0.0011407, 0.0011416
6: -0.0020709, -0.0010441, -0.0020780, -0.0010285, -0.0010424, 0.0010339
7: -0.0082211, -0.0075554, -0.0082324, -0.0075494, -0.0006716, 0.0006770
8: 0.0055508, 0.0094174, 0.0054717, 0.0094441, -0.0038330, 0.0038898
9: -0.0036822, -0.0032244, -0.0036828, -0.0032159, -0.0004663, 0.0004584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005674, upper bound: 0.0005491
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005674, upper bound: 0.0005491
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007830, 0.0011155, 0.0007980, 0.0011151, -0.0003321, 0.0003174
1: 0.9934286, 0.9942212, 0.9934336, 0.9941893, -0.0007607, 0.0007876
2: -0.0085713, -0.0057340, -0.0085194, -0.0055538, -0.0027958, 0.0025636
3: 0.0036677, 0.0041471, 0.0036865, 0.0041437, -0.0004760, 0.0004606
4: 0.0029488, 0.0051913, 0.0028064, 0.0051503, -0.0022014, 0.0023849
5: 0.0052178, 0.0063570, 0.0052534, 0.0063959, -0.0011781, 0.0011035
6: -0.0020963, -0.0009978, -0.0020782, -0.0010308, -0.0010655, 0.0010804
7: -0.0081873, -0.0075340, -0.0082214, -0.0075492, -0.0006381, 0.0006794
8: 0.0057853, 0.0095133, 0.0055486, 0.0094451, -0.0036001, 0.0039049
9: -0.0036804, -0.0031938, -0.0036822, -0.0032155, -0.0004648, 0.0004884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006383, upper bound: 0.0006278
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006383, upper bound: 0.0006278
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007769, 0.0011156, 0.0007980, 0.0011151, -0.0003381, 0.0003176
1: 0.9934266, 0.9942339, 0.9934336, 0.9941893, -0.0007627, 0.0008003
2: -0.0085922, -0.0057195, -0.0085194, -0.0055538, -0.0028141, 0.0025779
3: 0.0036601, 0.0041485, 0.0036865, 0.0041437, -0.0004836, 0.0004620
4: 0.0029374, 0.0052078, 0.0028064, 0.0051503, -0.0022129, 0.0024014
5: 0.0052035, 0.0063601, 0.0052534, 0.0063959, -0.0011924, 0.0011066
6: -0.0021035, -0.0009846, -0.0020782, -0.0010308, -0.0010728, 0.0010937
7: -0.0081900, -0.0075278, -0.0082214, -0.0075492, -0.0006408, 0.0006896
8: 0.0057664, 0.0095407, 0.0055486, 0.0094451, -0.0036191, 0.0039318
9: -0.0036805, -0.0031850, -0.0036822, -0.0032155, -0.0004650, 0.0004972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006383, upper bound: 0.0006437
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006383, upper bound: 0.0006278
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007830, 0.0011155, 0.0008263, 0.0011143, -0.0003313, 0.0002892
1: 0.9934286, 0.9942212, 0.9934430, 0.9941294, -0.0007008, 0.0007782
2: -0.0085713, -0.0057340, -0.0084218, -0.0054207, -0.0029283, 0.0024690
3: 0.0036677, 0.0041471, 0.0037218, 0.0041372, -0.0004696, 0.0004253
4: 0.0029488, 0.0051913, 0.0027012, 0.0050732, -0.0021243, 0.0024901
5: 0.0052178, 0.0063570, 0.0053204, 0.0064247, -0.0012069, 0.0010366
6: -0.0020963, -0.0009978, -0.0020444, -0.0010027, -0.0010936, 0.0010466
7: -0.0081873, -0.0075340, -0.0082465, -0.0075779, -0.0006094, 0.0007015
8: 0.0057853, 0.0095133, 0.0053738, 0.0093169, -0.0034718, 0.0040798
9: -0.0036804, -0.0031938, -0.0036836, -0.0032564, -0.0004239, 0.0004898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005647, upper bound: 0.0005645
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005609, upper bound: 0.0005609
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007769, 0.0011156, 0.0008263, 0.0011143, -0.0003374, 0.0002894
1: 0.9934266, 0.9942339, 0.9934430, 0.9941294, -0.0007028, 0.0007910
2: -0.0085922, -0.0057195, -0.0084218, -0.0054207, -0.0029466, 0.0024833
3: 0.0036601, 0.0041485, 0.0037218, 0.0041372, -0.0004771, 0.0004267
4: 0.0029374, 0.0052078, 0.0027012, 0.0050732, -0.0021357, 0.0025066
5: 0.0052035, 0.0063601, 0.0053204, 0.0064247, -0.0012212, 0.0010397
6: -0.0021035, -0.0009846, -0.0020444, -0.0010027, -0.0011009, 0.0010598
7: -0.0081900, -0.0075278, -0.0082465, -0.0075779, -0.0006122, 0.0007117
8: 0.0057664, 0.0095407, 0.0053738, 0.0093169, -0.0034908, 0.0041066
9: -0.0036805, -0.0031850, -0.0036836, -0.0032564, -0.0004241, 0.0004985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005647, upper bound: 0.0005645
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005609, upper bound: 0.0005609
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007887, 0.0011153, 0.0007980, 0.0011151, -0.0003264, 0.0003174
1: 0.9934306, 0.9942090, 0.9934335, 0.9941894, -0.0007589, 0.0007755
2: -0.0085515, -0.0057014, -0.0085197, -0.0055394, -0.0027871, 0.0025948
3: 0.0036748, 0.0041458, 0.0036864, 0.0041437, -0.0004689, 0.0004595
4: 0.0029230, 0.0051757, 0.0027950, 0.0051505, -0.0022275, 0.0023806
5: 0.0052314, 0.0063640, 0.0052532, 0.0063990, -0.0011676, 0.0011108
6: -0.0020894, -0.0010104, -0.0020783, -0.0010306, -0.0010588, 0.0010680
7: -0.0081935, -0.0075398, -0.0082241, -0.0075491, -0.0006443, 0.0006804
8: 0.0057425, 0.0094873, 0.0055297, 0.0094455, -0.0036428, 0.0038973
9: -0.0036807, -0.0032021, -0.0036824, -0.0032154, -0.0004653, 0.0004803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006389, upper bound: 0.0006389
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006389, upper bound: 0.0006389
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007830, 0.0011155, 0.0007980, 0.0011151, -0.0003321, 0.0003175
1: 0.9934287, 0.9942210, 0.9934335, 0.9941894, -0.0007607, 0.0007875
2: -0.0085712, -0.0056864, -0.0085197, -0.0055394, -0.0028059, 0.0026089
3: 0.0036677, 0.0041471, 0.0036864, 0.0041437, -0.0004760, 0.0004608
4: 0.0029112, 0.0051913, 0.0027950, 0.0051505, -0.0022393, 0.0023962
5: 0.0052178, 0.0063672, 0.0052532, 0.0063990, -0.0011812, 0.0011140
6: -0.0020962, -0.0009979, -0.0020783, -0.0010306, -0.0010657, 0.0010805
7: -0.0081963, -0.0075340, -0.0082241, -0.0075491, -0.0006471, 0.0006880
8: 0.0057228, 0.0095132, 0.0055297, 0.0094455, -0.0036621, 0.0039227
9: -0.0036809, -0.0031938, -0.0036824, -0.0032154, -0.0004654, 0.0004885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006389, upper bound: 0.0006559
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006389, upper bound: 0.0006631
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007887, 0.0011153, 0.0008262, 0.0011144, -0.0003256, 0.0002891
1: 0.9934306, 0.9942090, 0.9934429, 0.9941295, -0.0006990, 0.0007661
2: -0.0085515, -0.0057014, -0.0084221, -0.0054071, -0.0029192, 0.0025013
3: 0.0036748, 0.0041458, 0.0037217, 0.0041373, -0.0004624, 0.0004241
4: 0.0029230, 0.0051757, 0.0026905, 0.0050734, -0.0021504, 0.0024852
5: 0.0052314, 0.0063640, 0.0053201, 0.0064276, -0.0011962, 0.0010439
6: -0.0020894, -0.0010104, -0.0020445, -0.0009979, -0.0010915, 0.0010341
7: -0.0081935, -0.0075398, -0.0082491, -0.0075778, -0.0006157, 0.0007024
8: 0.0057425, 0.0094873, 0.0053559, 0.0093173, -0.0035148, 0.0040712
9: -0.0036807, -0.0032021, -0.0036837, -0.0032563, -0.0004244, 0.0004816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005660
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0006131
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007830, 0.0011155, 0.0008262, 0.0011144, -0.0003313, 0.0002893
1: 0.9934287, 0.9942210, 0.9934429, 0.9941295, -0.0007008, 0.0007781
2: -0.0085712, -0.0056864, -0.0084221, -0.0054071, -0.0029380, 0.0025155
3: 0.0036677, 0.0041471, 0.0037217, 0.0041373, -0.0004696, 0.0004254
4: 0.0029112, 0.0051913, 0.0026905, 0.0050734, -0.0021622, 0.0025008
5: 0.0052178, 0.0063672, 0.0053201, 0.0064276, -0.0012098, 0.0010471
6: -0.0020962, -0.0009979, -0.0020445, -0.0009979, -0.0010983, 0.0010466
7: -0.0081963, -0.0075340, -0.0082491, -0.0075778, -0.0006185, 0.0007100
8: 0.0057228, 0.0095132, 0.0053559, 0.0093173, -0.0035341, 0.0040967
9: -0.0036809, -0.0031938, -0.0036837, -0.0032563, -0.0004245, 0.0004899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005730
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0006355
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007830, 0.0011155, 0.0007906, 0.0011153, -0.0003323, 0.0003248
1: 0.9934286, 0.9942212, 0.9934312, 0.9942050, -0.0007764, 0.0007900
2: -0.0085713, -0.0057340, -0.0085449, -0.0055413, -0.0028123, 0.0025953
3: 0.0036677, 0.0041471, 0.0036772, 0.0041454, -0.0004777, 0.0004699
4: 0.0029488, 0.0051913, 0.0027966, 0.0051705, -0.0022216, 0.0023948
5: 0.0052178, 0.0063570, 0.0052359, 0.0063986, -0.0011808, 0.0011210
6: -0.0020963, -0.0009978, -0.0020871, -0.0010146, -0.0010817, 0.0010893
7: -0.0081873, -0.0075340, -0.0082237, -0.0075417, -0.0006456, 0.0006849
8: 0.0057853, 0.0095133, 0.0055322, 0.0094786, -0.0036347, 0.0039223
9: -0.0036804, -0.0031938, -0.0036823, -0.0032049, -0.0004755, 0.0004885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006383, upper bound: 0.0006278
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006383, upper bound: 0.0006278
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007769, 0.0011156, 0.0007906, 0.0011153, -0.0003383, 0.0003250
1: 0.9934266, 0.9942339, 0.9934312, 0.9942050, -0.0007784, 0.0008028
2: -0.0085922, -0.0057195, -0.0085449, -0.0055413, -0.0028277, 0.0026060
3: 0.0036601, 0.0041485, 0.0036772, 0.0041454, -0.0004853, 0.0004713
4: 0.0029374, 0.0052078, 0.0027966, 0.0051705, -0.0022331, 0.0024113
5: 0.0052035, 0.0063601, 0.0052359, 0.0063986, -0.0011951, 0.0011242
6: -0.0021035, -0.0009846, -0.0020871, -0.0010146, -0.0010890, 0.0011025
7: -0.0081900, -0.0075278, -0.0082237, -0.0075417, -0.0006483, 0.0006858
8: 0.0057664, 0.0095407, 0.0055322, 0.0094786, -0.0036530, 0.0039482
9: -0.0036805, -0.0031850, -0.0036823, -0.0032049, -0.0004757, 0.0004973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006383, upper bound: 0.0006278
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006383, upper bound: 0.0006535
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007830, 0.0011155, 0.0008183, 0.0011146, -0.0003316, 0.0002972
1: 0.9934286, 0.9942212, 0.9934403, 0.9941465, -0.0007179, 0.0007809
2: -0.0085713, -0.0057340, -0.0084495, -0.0054051, -0.0029491, 0.0025016
3: 0.0036677, 0.0041471, 0.0037118, 0.0041391, -0.0004714, 0.0004353
4: 0.0029488, 0.0051913, 0.0026889, 0.0050950, -0.0021462, 0.0025024
5: 0.0052178, 0.0063570, 0.0053014, 0.0064280, -0.0012103, 0.0010555
6: -0.0020963, -0.0009978, -0.0020540, -0.0009972, -0.0010990, 0.0010561
7: -0.0081873, -0.0075340, -0.0082495, -0.0075697, -0.0006176, 0.0007070
8: 0.0057853, 0.0095133, 0.0053532, 0.0093532, -0.0035103, 0.0041008
9: -0.0036804, -0.0031938, -0.0036837, -0.0032449, -0.0004355, 0.0004899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005634, upper bound: 0.0005633
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005609, upper bound: 0.0005610
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007769, 0.0011156, 0.0008183, 0.0011146, -0.0003376, 0.0002973
1: 0.9934266, 0.9942339, 0.9934403, 0.9941465, -0.0007199, 0.0007936
2: -0.0085922, -0.0057195, -0.0084495, -0.0054051, -0.0029635, 0.0025123
3: 0.0036601, 0.0041485, 0.0037118, 0.0041391, -0.0004790, 0.0004367
4: 0.0029374, 0.0052078, 0.0026889, 0.0050950, -0.0021576, 0.0025189
5: 0.0052035, 0.0063601, 0.0053014, 0.0064280, -0.0012246, 0.0010587
6: -0.0021035, -0.0009846, -0.0020540, -0.0009972, -0.0011063, 0.0010694
7: -0.0081900, -0.0075278, -0.0082495, -0.0075697, -0.0006203, 0.0007079
8: 0.0057664, 0.0095407, 0.0053532, 0.0093532, -0.0035284, 0.0041268
9: -0.0036805, -0.0031850, -0.0036837, -0.0032449, -0.0004357, 0.0004987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005634, upper bound: 0.0006036
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005609, upper bound: 0.0006030
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007887, 0.0011153, 0.0007906, 0.0011153, -0.0003265, 0.0003248
1: 0.9934306, 0.9942090, 0.9934311, 0.9942052, -0.0007746, 0.0007779
2: -0.0085515, -0.0057014, -0.0085453, -0.0055264, -0.0028044, 0.0026278
3: 0.0036748, 0.0041458, 0.0036771, 0.0041454, -0.0004706, 0.0004687
4: 0.0029230, 0.0051757, 0.0027847, 0.0051707, -0.0022477, 0.0023909
5: 0.0052314, 0.0063640, 0.0052357, 0.0064018, -0.0011705, 0.0011283
6: -0.0020894, -0.0010104, -0.0020872, -0.0010144, -0.0010750, 0.0010769
7: -0.0081935, -0.0075398, -0.0082266, -0.0075416, -0.0006518, 0.0006860
8: 0.0057425, 0.0094873, 0.0055126, 0.0094790, -0.0036776, 0.0039154
9: -0.0036807, -0.0032021, -0.0036825, -0.0032047, -0.0004760, 0.0004804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006389, upper bound: 0.0006389
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006389, upper bound: 0.0006389
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007830, 0.0011155, 0.0007906, 0.0011153, -0.0003323, 0.0003249
1: 0.9934287, 0.9942210, 0.9934311, 0.9942052, -0.0007765, 0.0007899
2: -0.0085712, -0.0056864, -0.0085453, -0.0055264, -0.0028199, 0.0026382
3: 0.0036677, 0.0041471, 0.0036771, 0.0041454, -0.0004777, 0.0004700
4: 0.0029112, 0.0051913, 0.0027847, 0.0051707, -0.0022595, 0.0024065
5: 0.0052178, 0.0063672, 0.0052357, 0.0064018, -0.0011840, 0.0011316
6: -0.0020962, -0.0009979, -0.0020872, -0.0010144, -0.0010819, 0.0010894
7: -0.0081963, -0.0075340, -0.0082266, -0.0075416, -0.0006546, 0.0006856
8: 0.0057228, 0.0095132, 0.0055126, 0.0094790, -0.0036962, 0.0039398
9: -0.0036809, -0.0031938, -0.0036825, -0.0032047, -0.0004761, 0.0004887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006389, upper bound: 0.0006631
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006389, upper bound: 0.0006631
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007887, 0.0011153, 0.0008182, 0.0011146, -0.0003258, 0.0002971
1: 0.9934306, 0.9942090, 0.9934402, 0.9941465, -0.0007160, 0.0007688
2: -0.0085515, -0.0057014, -0.0084497, -0.0053918, -0.0029391, 0.0025350
3: 0.0036748, 0.0041458, 0.0037117, 0.0041391, -0.0004642, 0.0004341
4: 0.0029230, 0.0051757, 0.0026783, 0.0050952, -0.0021722, 0.0024973
5: 0.0052314, 0.0063640, 0.0053012, 0.0064309, -0.0011996, 0.0010628
6: -0.0020894, -0.0010104, -0.0020541, -0.0009926, -0.0010968, 0.0010437
7: -0.0081935, -0.0075398, -0.0082520, -0.0075697, -0.0006238, 0.0007079
8: 0.0057425, 0.0094873, 0.0053357, 0.0093535, -0.0035534, 0.0040913
9: -0.0036807, -0.0032021, -0.0036839, -0.0032447, -0.0004360, 0.0004818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005991
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005991
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007830, 0.0011155, 0.0008182, 0.0011146, -0.0003315, 0.0002973
1: 0.9934287, 0.9942210, 0.9934402, 0.9941465, -0.0007178, 0.0007808
2: -0.0085712, -0.0056864, -0.0084497, -0.0053918, -0.0029539, 0.0025454
3: 0.0036677, 0.0041471, 0.0037117, 0.0041391, -0.0004714, 0.0004354
4: 0.0029112, 0.0051913, 0.0026783, 0.0050952, -0.0021840, 0.0025129
5: 0.0052178, 0.0063672, 0.0053012, 0.0064309, -0.0012131, 0.0010660
6: -0.0020962, -0.0009979, -0.0020541, -0.0009926, -0.0011036, 0.0010562
7: -0.0081963, -0.0075340, -0.0082520, -0.0075697, -0.0006266, 0.0007075
8: 0.0057228, 0.0095132, 0.0053357, 0.0093535, -0.0035717, 0.0041161
9: -0.0036809, -0.0031938, -0.0036839, -0.0032447, -0.0004361, 0.0004900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005660
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0006428
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0008107, 0.0011148, 0.0007763, 0.0011157, -0.0003049, 0.0003385
1: 0.9934377, 0.9941624, 0.9934264, 0.9942352, -0.0007975, 0.0007359
2: -0.0084756, -0.0055531, -0.0085945, -0.0056937, -0.0025678, 0.0028129
3: 0.0037024, 0.0041408, 0.0036593, 0.0041487, -0.0004463, 0.0004815
4: 0.0028059, 0.0051156, 0.0029170, 0.0052096, -0.0024037, 0.0021986
5: 0.0052835, 0.0063960, 0.0052019, 0.0063656, -0.0010822, 0.0011941
6: -0.0020630, -0.0010486, -0.0021043, -0.0009831, -0.0010799, 0.0010557
7: -0.0082215, -0.0075621, -0.0081949, -0.0075272, -0.0006934, 0.0006328
8: 0.0055478, 0.0093875, 0.0057325, 0.0095437, -0.0039348, 0.0035979
9: -0.0036822, -0.0032339, -0.0036808, -0.0031841, -0.0004981, 0.0004469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006083
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006083
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0008041, 0.0011149, 0.0007763, 0.0011157, -0.0003115, 0.0003386
1: 0.9934356, 0.9941763, 0.9934264, 0.9942352, -0.0007996, 0.0007499
2: -0.0084983, -0.0055554, -0.0085945, -0.0056937, -0.0025897, 0.0028071
3: 0.0036941, 0.0041423, 0.0036593, 0.0041487, -0.0004545, 0.0004830
4: 0.0028077, 0.0051336, 0.0029170, 0.0052096, -0.0024019, 0.0022166
5: 0.0052679, 0.0063955, 0.0052019, 0.0063656, -0.0010978, 0.0011936
6: -0.0020709, -0.0010441, -0.0021043, -0.0009831, -0.0010878, 0.0010602
7: -0.0082211, -0.0075554, -0.0081949, -0.0075272, -0.0006906, 0.0006395
8: 0.0055508, 0.0094174, 0.0057325, 0.0095437, -0.0039310, 0.0036273
9: -0.0036822, -0.0032244, -0.0036808, -0.0031841, -0.0004981, 0.0004564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006083
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006083
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0008061, 0.0011149, 0.0007729, 0.0011157, -0.0003096, 0.0003420
1: 0.9934363, 0.9941722, 0.9934254, 0.9942424, -0.0008062, 0.0007468
2: -0.0084915, -0.0055130, -0.0086062, -0.0056797, -0.0025836, 0.0028673
3: 0.0036966, 0.0041419, 0.0036550, 0.0041494, -0.0004529, 0.0004868
4: 0.0027742, 0.0051283, 0.0029060, 0.0052189, -0.0024447, 0.0022223
5: 0.0052725, 0.0064047, 0.0051939, 0.0063687, -0.0010962, 0.0012108
6: -0.0020686, -0.0010347, -0.0021084, -0.0009757, -0.0010929, 0.0010737
7: -0.0082291, -0.0075574, -0.0081976, -0.0075238, -0.0007012, 0.0006402
8: 0.0054950, 0.0094085, 0.0057140, 0.0095591, -0.0040036, 0.0036334
9: -0.0036826, -0.0032272, -0.0036809, -0.0031792, -0.0005035, 0.0004537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006366, upper bound: 0.0006230
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006630, upper bound: 0.0006419
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0008347, 0.0011141, 0.0007729, 0.0011157, -0.0002810, 0.0003412
1: 0.9934456, 0.9941114, 0.9934254, 0.9942424, -0.0007968, 0.0006860
2: -0.0083925, -0.0053817, -0.0086062, -0.0056797, -0.0024846, 0.0029989
3: 0.0037324, 0.0041353, 0.0036550, 0.0041494, -0.0004170, 0.0004803
4: 0.0026704, 0.0050500, 0.0029060, 0.0052189, -0.0025485, 0.0021441
5: 0.0053404, 0.0064331, 0.0051939, 0.0063687, -0.0010282, 0.0012392
6: -0.0020342, -0.0009891, -0.0021084, -0.0009757, -0.0010585, 0.0011193
7: -0.0082539, -0.0075865, -0.0081976, -0.0075238, -0.0007239, 0.0006111
8: 0.0053225, 0.0092784, 0.0057140, 0.0095591, -0.0041762, 0.0035033
9: -0.0036840, -0.0032687, -0.0036809, -0.0031792, -0.0005048, 0.0004122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006366, upper bound: 0.0006230
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006630, upper bound: 0.0006419
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0008061, 0.0011149, 0.0007669, 0.0011159, -0.0003098, 0.0003480
1: 0.9934363, 0.9941722, 0.9934233, 0.9942552, -0.0008189, 0.0007489
2: -0.0084915, -0.0055130, -0.0086270, -0.0056870, -0.0025788, 0.0028923
3: 0.0036966, 0.0041419, 0.0036475, 0.0041508, -0.0004543, 0.0004943
4: 0.0027742, 0.0051283, 0.0029117, 0.0052353, -0.0024611, 0.0022166
5: 0.0052725, 0.0064047, 0.0051796, 0.0063671, -0.0010946, 0.0012251
6: -0.0020686, -0.0010347, -0.0021156, -0.0009625, -0.0011061, 0.0010809
7: -0.0082291, -0.0075574, -0.0081962, -0.0075177, -0.0007070, 0.0006388
8: 0.0054950, 0.0094085, 0.0057236, 0.0095865, -0.0040317, 0.0036246
9: -0.0036826, -0.0032272, -0.0036808, -0.0031704, -0.0005122, 0.0004536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006037, upper bound: 0.0005982
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006354, upper bound: 0.0006256
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0008347, 0.0011141, 0.0007669, 0.0011159, -0.0002812, 0.0003473
1: 0.9934456, 0.9941114, 0.9934233, 0.9942552, -0.0008096, 0.0006881
2: -0.0083925, -0.0053817, -0.0086270, -0.0056870, -0.0024798, 0.0030239
3: 0.0037324, 0.0041353, 0.0036475, 0.0041508, -0.0004184, 0.0004878
4: 0.0026704, 0.0050500, 0.0029117, 0.0052353, -0.0025649, 0.0021383
5: 0.0053404, 0.0064331, 0.0051796, 0.0063671, -0.0010267, 0.0012535
6: -0.0020342, -0.0009891, -0.0021156, -0.0009625, -0.0010717, 0.0011265
7: -0.0082539, -0.0075865, -0.0081962, -0.0075177, -0.0007296, 0.0006097
8: 0.0053225, 0.0092784, 0.0057236, 0.0095865, -0.0042043, 0.0034945
9: -0.0036840, -0.0032687, -0.0036808, -0.0031704, -0.0005135, 0.0004121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006037, upper bound: 0.0005982
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006354, upper bound: 0.0006256
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007986, 0.0011151, 0.0007711, 0.0011158, -0.0003172, 0.0003440
1: 0.9934338, 0.9941880, 0.9934248, 0.9942463, -0.0008125, 0.0007632
2: -0.0085174, -0.0054947, -0.0086124, -0.0056777, -0.0026125, 0.0028960
3: 0.0036872, 0.0041436, 0.0036528, 0.0041499, -0.0004626, 0.0004908
4: 0.0027598, 0.0051487, 0.0029044, 0.0052238, -0.0024641, 0.0022443
5: 0.0052548, 0.0064087, 0.0051896, 0.0063691, -0.0011143, 0.0012191
6: -0.0020776, -0.0010284, -0.0021105, -0.0009717, -0.0011058, 0.0010822
7: -0.0082325, -0.0075498, -0.0081979, -0.0075219, -0.0007104, 0.0006481
8: 0.0054710, 0.0094425, 0.0057114, 0.0095673, -0.0040368, 0.0036701
9: -0.0036828, -0.0032164, -0.0036809, -0.0031766, -0.0005063, 0.0004645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006374, upper bound: 0.0006245
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006668, upper bound: 0.0006465
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0008281, 0.0011143, 0.0007711, 0.0011158, -0.0002877, 0.0003432
1: 0.9934435, 0.9941255, 0.9934248, 0.9942463, -0.0008028, 0.0007008
2: -0.0084156, -0.0053639, -0.0086124, -0.0056777, -0.0025130, 0.0030292
3: 0.0037241, 0.0041368, 0.0036528, 0.0041499, -0.0004258, 0.0004840
4: 0.0026563, 0.0050683, 0.0029044, 0.0052238, -0.0025675, 0.0021639
5: 0.0053246, 0.0064369, 0.0051896, 0.0063691, -0.0010445, 0.0012473
6: -0.0020422, -0.0009830, -0.0021105, -0.0009717, -0.0010705, 0.0011276
7: -0.0082573, -0.0075797, -0.0081979, -0.0075219, -0.0007317, 0.0006183
8: 0.0052991, 0.0093087, 0.0057114, 0.0095673, -0.0042078, 0.0035367
9: -0.0036842, -0.0032590, -0.0036809, -0.0031766, -0.0005076, 0.0004219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006374, upper bound: 0.0006245
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006668, upper bound: 0.0006465
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007986, 0.0011151, 0.0007651, 0.0011159, -0.0003173, 0.0003500
1: 0.9934338, 0.9941880, 0.9934227, 0.9942590, -0.0008252, 0.0007653
2: -0.0085174, -0.0054947, -0.0086331, -0.0056850, -0.0026078, 0.0029202
3: 0.0036872, 0.0041436, 0.0036453, 0.0041512, -0.0004640, 0.0004982
4: 0.0027598, 0.0051487, 0.0029101, 0.0052401, -0.0024804, 0.0022386
5: 0.0052548, 0.0064087, 0.0051755, 0.0063675, -0.0011128, 0.0012332
6: -0.0020776, -0.0010284, -0.0021177, -0.0009586, -0.0011189, 0.0010893
7: -0.0082325, -0.0075498, -0.0081966, -0.0075159, -0.0007163, 0.0006468
8: 0.0054710, 0.0094425, 0.0057209, 0.0095944, -0.0040649, 0.0036614
9: -0.0036828, -0.0032164, -0.0036809, -0.0031679, -0.0005149, 0.0004645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006080, upper bound: 0.0006034
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006429, upper bound: 0.0006351
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0008281, 0.0011143, 0.0007651, 0.0011159, -0.0002879, 0.0003492
1: 0.9934435, 0.9941255, 0.9934227, 0.9942590, -0.0008155, 0.0007028
2: -0.0084156, -0.0053639, -0.0086331, -0.0056850, -0.0025083, 0.0030534
3: 0.0037241, 0.0041368, 0.0036453, 0.0041512, -0.0004271, 0.0004915
4: 0.0026563, 0.0050683, 0.0029101, 0.0052401, -0.0025838, 0.0021582
5: 0.0053246, 0.0064369, 0.0051755, 0.0063675, -0.0010429, 0.0012615
6: -0.0020422, -0.0009830, -0.0021177, -0.0009586, -0.0010836, 0.0011347
7: -0.0082573, -0.0075797, -0.0081966, -0.0075159, -0.0007376, 0.0006169
8: 0.0052991, 0.0093087, 0.0057209, 0.0095944, -0.0042359, 0.0035281
9: -0.0036842, -0.0032590, -0.0036809, -0.0031679, -0.0005162, 0.0004218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006080, upper bound: 0.0006034
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006429, upper bound: 0.0006351
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0008059, 0.0011149, 0.0008013, 0.0011150, -0.0003091, 0.0003135
1: 0.9934362, 0.9941726, 0.9934347, 0.9941822, -0.0007460, 0.0007379
2: -0.0084922, -0.0055512, -0.0085079, -0.0055493, -0.0027162, 0.0027467
3: 0.0036964, 0.0041419, 0.0036907, 0.0041429, -0.0004466, 0.0004512
4: 0.0028044, 0.0051288, 0.0028029, 0.0051412, -0.0023368, 0.0023259
5: 0.0052721, 0.0063965, 0.0052613, 0.0063969, -0.0011248, 0.0011352
6: -0.0020688, -0.0010480, -0.0020743, -0.0010380, -0.0010308, 0.0010263
7: -0.0082219, -0.0075572, -0.0082222, -0.0075526, -0.0006693, 0.0006650
8: 0.0055452, 0.0094093, 0.0055427, 0.0094300, -0.0038284, 0.0038061
9: -0.0036822, -0.0032270, -0.0036823, -0.0032204, -0.0004619, 0.0004553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006279
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006367
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0008059, 0.0011149, 0.0007947, 0.0011152, -0.0003093, 0.0003201
1: 0.9934362, 0.9941726, 0.9934325, 0.9941962, -0.0007600, 0.0007401
2: -0.0084922, -0.0055512, -0.0085308, -0.0055515, -0.0027136, 0.0027732
3: 0.0036964, 0.0041419, 0.0036823, 0.0041444, -0.0004481, 0.0004595
4: 0.0028044, 0.0051288, 0.0028046, 0.0051592, -0.0023549, 0.0023242
5: 0.0052721, 0.0063965, 0.0052456, 0.0063964, -0.0011243, 0.0011508
6: -0.0020688, -0.0010480, -0.0020822, -0.0010236, -0.0010452, 0.0010342
7: -0.0082219, -0.0075572, -0.0082218, -0.0075459, -0.0006760, 0.0006646
8: 0.0055452, 0.0094093, 0.0055456, 0.0094600, -0.0038595, 0.0038031
9: -0.0036822, -0.0032270, -0.0036822, -0.0032108, -0.0004714, 0.0004553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005674
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005674
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0007988, 0.0011151, 0.0007995, 0.0011151, -0.0003163, 0.0003156
1: 0.9934339, 0.9941877, 0.9934341, 0.9941861, -0.0007522, 0.0007536
2: -0.0085167, -0.0055312, -0.0085144, -0.0055473, -0.0027439, 0.0027764
3: 0.0036874, 0.0041435, 0.0036883, 0.0041434, -0.0004559, 0.0004552
4: 0.0027886, 0.0051482, 0.0028013, 0.0051463, -0.0023577, 0.0023469
5: 0.0052552, 0.0064008, 0.0052568, 0.0063973, -0.0011421, 0.0011439
6: -0.0020773, -0.0010324, -0.0020765, -0.0010339, -0.0010434, 0.0010441
7: -0.0082256, -0.0075500, -0.0082226, -0.0075507, -0.0006749, 0.0006726
8: 0.0055190, 0.0094416, 0.0055401, 0.0094385, -0.0038638, 0.0038413
9: -0.0036824, -0.0032166, -0.0036823, -0.0032176, -0.0004648, 0.0004656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006288
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006439
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0007988, 0.0011151, 0.0007930, 0.0011152, -0.0003164, 0.0003221
1: 0.9934339, 0.9941877, 0.9934319, 0.9941999, -0.0007660, 0.0007558
2: -0.0085167, -0.0055312, -0.0085368, -0.0055495, -0.0027413, 0.0028024
3: 0.0036874, 0.0041435, 0.0036802, 0.0041448, -0.0004574, 0.0004634
4: 0.0027886, 0.0051482, 0.0028030, 0.0051640, -0.0023754, 0.0023452
5: 0.0052552, 0.0064008, 0.0052415, 0.0063968, -0.0011416, 0.0011593
6: -0.0020773, -0.0010324, -0.0020843, -0.0010197, -0.0010576, 0.0010519
7: -0.0082256, -0.0075500, -0.0082222, -0.0075441, -0.0006815, 0.0006722
8: 0.0055190, 0.0094416, 0.0055430, 0.0094679, -0.0038942, 0.0038383
9: -0.0036824, -0.0032166, -0.0036823, -0.0032083, -0.0004742, 0.0004656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005678
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005678
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007976, 0.0011151, 0.0007677, 0.0011159, -0.0003183, 0.0003474
1: 0.9934334, 0.9941902, 0.9934236, 0.9942535, -0.0008201, 0.0007665
2: -0.0085209, -0.0055236, -0.0086242, -0.0057134, -0.0025855, 0.0028786
3: 0.0036859, 0.0041438, 0.0036485, 0.0041506, -0.0004647, 0.0004953
4: 0.0027826, 0.0051515, 0.0029325, 0.0052331, -0.0024505, 0.0022189
5: 0.0052524, 0.0064024, 0.0051815, 0.0063614, -0.0011090, 0.0012209
6: -0.0020788, -0.0010298, -0.0021146, -0.0009643, -0.0011145, 0.0010848
7: -0.0082271, -0.0075488, -0.0081912, -0.0075185, -0.0006968, 0.0006424
8: 0.0055090, 0.0094470, 0.0057583, 0.0095828, -0.0040143, 0.0036292
9: -0.0036825, -0.0032149, -0.0036806, -0.0031716, -0.0005109, 0.0004656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005852, upper bound: 0.0005836
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005852, upper bound: 0.0005836
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0008259, 0.0011144, 0.0007677, 0.0011159, -0.0002900, 0.0003467
1: 0.9934427, 0.9941303, 0.9934236, 0.9942535, -0.0008108, 0.0007067
2: -0.0084231, -0.0053892, -0.0086242, -0.0057134, -0.0024907, 0.0030152
3: 0.0037213, 0.0041373, 0.0036485, 0.0041506, -0.0004293, 0.0004888
4: 0.0026763, 0.0050742, 0.0029325, 0.0052331, -0.0025568, 0.0021417
5: 0.0053195, 0.0064315, 0.0051815, 0.0063614, -0.0010419, 0.0012500
6: -0.0020448, -0.0009917, -0.0021146, -0.0009643, -0.0010806, 0.0011229
7: -0.0082525, -0.0075775, -0.0081912, -0.0075185, -0.0007199, 0.0006137
8: 0.0053323, 0.0093186, 0.0057583, 0.0095828, -0.0041910, 0.0035006
9: -0.0036839, -0.0032559, -0.0036806, -0.0031716, -0.0005123, 0.0004247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005852, upper bound: 0.0005836
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005852, upper bound: 0.0005979
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0007975, 0.0011151, 0.0007736, 0.0011157, -0.0003182, 0.0003415
1: 0.9934334, 0.9941904, 0.9934255, 0.9942411, -0.0008077, 0.0007648
2: -0.0085212, -0.0055093, -0.0086039, -0.0056800, -0.0026168, 0.0028695
3: 0.0036858, 0.0041438, 0.0036559, 0.0041493, -0.0004635, 0.0004879
4: 0.0027712, 0.0051517, 0.0029062, 0.0052171, -0.0024459, 0.0022455
5: 0.0052522, 0.0064055, 0.0051954, 0.0063686, -0.0011164, 0.0012101
6: -0.0020789, -0.0010296, -0.0021076, -0.0009771, -0.0011018, 0.0010780
7: -0.0082298, -0.0075487, -0.0081975, -0.0075244, -0.0006962, 0.0006488
8: 0.0054901, 0.0094474, 0.0057145, 0.0095562, -0.0040057, 0.0036726
9: -0.0036827, -0.0032148, -0.0036809, -0.0031801, -0.0005026, 0.0004661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006131, upper bound: 0.0006120
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006131, upper bound: 0.0006229
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0008258, 0.0011144, 0.0007736, 0.0011157, -0.0002899, 0.0003408
1: 0.9934427, 0.9941304, 0.9934255, 0.9942411, -0.0007984, 0.0007048
2: -0.0084234, -0.0053760, -0.0086039, -0.0056800, -0.0025231, 0.0030052
3: 0.0037212, 0.0041373, 0.0036559, 0.0041493, -0.0004280, 0.0004815
4: 0.0026659, 0.0050745, 0.0029062, 0.0052171, -0.0025512, 0.0021683
5: 0.0053192, 0.0064343, 0.0051954, 0.0063686, -0.0010494, 0.0012389
6: -0.0020449, -0.0009871, -0.0021076, -0.0009771, -0.0010678, 0.0011205
7: -0.0082550, -0.0075774, -0.0081975, -0.0075244, -0.0007194, 0.0006201
8: 0.0053150, 0.0093190, 0.0057145, 0.0095562, -0.0041812, 0.0035443
9: -0.0036840, -0.0032558, -0.0036809, -0.0031801, -0.0005039, 0.0004251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006131, upper bound: 0.0006120
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0006131, upper bound: 0.0006228
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0007902, 0.0011153, 0.0007658, 0.0011159, -0.0003257, 0.0003495
1: 0.9934310, 0.9942059, 0.9934230, 0.9942575, -0.0008264, 0.0007828
2: -0.0085464, -0.0055043, -0.0086308, -0.0057112, -0.0026158, 0.0029112
3: 0.0036767, 0.0041455, 0.0036461, 0.0041511, -0.0004744, 0.0004993
4: 0.0027673, 0.0051716, 0.0029308, 0.0052383, -0.0024711, 0.0022408
5: 0.0052349, 0.0064066, 0.0051770, 0.0063619, -0.0011270, 0.0012296
6: -0.0020876, -0.0010136, -0.0021169, -0.0009601, -0.0011275, 0.0011033
7: -0.0082307, -0.0075413, -0.0081916, -0.0075165, -0.0007066, 0.0006503
8: 0.0054835, 0.0094805, 0.0057554, 0.0095914, -0.0040492, 0.0036659
9: -0.0036827, -0.0032042, -0.0036806, -0.0031689, -0.0005138, 0.0004764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005829, upper bound: 0.0005831
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005829, upper bound: 0.0005831
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0008179, 0.0011146, 0.0007658, 0.0011159, -0.0002980, 0.0003488
1: 0.9934401, 0.9941472, 0.9934230, 0.9942575, -0.0008174, 0.0007241
2: -0.0084508, -0.0053667, -0.0086308, -0.0057112, -0.0025220, 0.0030490
3: 0.0037113, 0.0041392, 0.0036461, 0.0041511, -0.0004397, 0.0004930
4: 0.0026585, 0.0050960, 0.0029308, 0.0052383, -0.0025798, 0.0021652
5: 0.0053005, 0.0064363, 0.0051770, 0.0063619, -0.0010614, 0.0012593
6: -0.0020544, -0.0009839, -0.0021169, -0.0009601, -0.0010943, 0.0011330
7: -0.0082568, -0.0075694, -0.0081916, -0.0075165, -0.0007286, 0.0006222
8: 0.0053027, 0.0093549, 0.0057554, 0.0095914, -0.0042308, 0.0035411
9: -0.0036841, -0.0032443, -0.0036806, -0.0031689, -0.0005153, 0.0004363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005829, upper bound: 0.0005831
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0005829, upper bound: 0.0006014
time: 0.56 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.06 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006655, upper bound: 0.0006678
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006655, upper bound: 0.0006678
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006688, upper bound: 0.0006787
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006688, upper bound: 0.0006787
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005903, upper bound: 0.0006004
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005903, upper bound: 0.0006004
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006392
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006630
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006567, upper bound: 0.0006479
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006567, upper bound: 0.0006479
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006609, upper bound: 0.0006585
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006609, upper bound: 0.0006585
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005930, upper bound: 0.0005990
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005914, upper bound: 0.0005979
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006026, upper bound: 0.0006092
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006026, upper bound: 0.0006354
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006781, upper bound: 0.0006718
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006781, upper bound: 0.0006718
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006873, upper bound: 0.0006872
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006873, upper bound: 0.0006872
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006100, upper bound: 0.0006191
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006100, upper bound: 0.0006374
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006367, upper bound: 0.0006599
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006367, upper bound: 0.0006668
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006732, upper bound: 0.0006535
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006732, upper bound: 0.0006535
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006785, upper bound: 0.0006631
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006785, upper bound: 0.0006631
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005803, upper bound: 0.0005796
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005803, upper bound: 0.0006080
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006310, upper bound: 0.0006344
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006310, upper bound: 0.0006429
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006054
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006054
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006241
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006241
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005742, upper bound: 0.0005765
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006279, upper bound: 0.0006054
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005742, upper bound: 0.0006048
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006279, upper bound: 0.0006365
IS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006054
IS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006054
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006367
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006367
IS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005634, upper bound: 0.0005635
IS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006279, upper bound: 0.0006054
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005634, upper bound: 0.0006042
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006279, upper bound: 0.0006367
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005674, upper bound: 0.0005491
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005674, upper bound: 0.0005491
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006383, upper bound: 0.0006278
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006383, upper bound: 0.0006278
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006383, upper bound: 0.0006437
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006383, upper bound: 0.0006278
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005647, upper bound: 0.0005645
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005609, upper bound: 0.0005609
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005647, upper bound: 0.0005645
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005609, upper bound: 0.0005609
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006389, upper bound: 0.0006389
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006389, upper bound: 0.0006389
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006389, upper bound: 0.0006559
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006389, upper bound: 0.0006631
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005660
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0006131
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005730
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0006355
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006383, upper bound: 0.0006278
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006383, upper bound: 0.0006278
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006383, upper bound: 0.0006278
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006383, upper bound: 0.0006535
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005634, upper bound: 0.0005633
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005609, upper bound: 0.0005610
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005634, upper bound: 0.0006036
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005609, upper bound: 0.0006030
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006389, upper bound: 0.0006389
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006389, upper bound: 0.0006389
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006389, upper bound: 0.0006631
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006389, upper bound: 0.0006631
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005991
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005991
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005660
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0006428
IS_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006083
IS_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006083
IS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006083
IS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006083, upper bound: 0.0006083
IS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006366, upper bound: 0.0006230
IS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006630, upper bound: 0.0006419
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006366, upper bound: 0.0006230
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006630, upper bound: 0.0006419
IS_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006037, upper bound: 0.0005982
IS_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006354, upper bound: 0.0006256
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006037, upper bound: 0.0005982
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006354, upper bound: 0.0006256
IS_A2_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006374, upper bound: 0.0006245
IS_A2_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006668, upper bound: 0.0006465
IS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006374, upper bound: 0.0006245
IS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006668, upper bound: 0.0006465
IS_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006080, upper bound: 0.0006034
IS_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006429, upper bound: 0.0006351
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006080, upper bound: 0.0006034
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006429, upper bound: 0.0006351
IS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006279
IS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006367
IS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005674
IS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005674
IS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006288
IS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006054, upper bound: 0.0006439
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005678
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005491, upper bound: 0.0005678
IS_A2_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005852, upper bound: 0.0005836
IS_A2_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005852, upper bound: 0.0005836
IS_A2_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005852, upper bound: 0.0005836
IS_A2_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005852, upper bound: 0.0005979
IS_A2_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006131, upper bound: 0.0006120
IS_A2_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006131, upper bound: 0.0006229
IS_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006131, upper bound: 0.0006120
IS_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0006131, upper bound: 0.0006228
IS_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005829, upper bound: 0.0005831
IS_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005829, upper bound: 0.0005831
IS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005829, upper bound: 0.0005831
IS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 1, lower bound: -0.0005829, upper bound: 0.0006014
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006428, upper bound: 0.0006322
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006428, upper bound: 0.0006322
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006077, upper bound: 0.0006146
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006077, upper bound: 0.0006146
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006602
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006658
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006602
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006658
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006387, upper bound: 0.0006339
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006471
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006387, upper bound: 0.0006339
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006472
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006647
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006672
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006647
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006672
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006450
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006477
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006450
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006450
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006600
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006650
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006600
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006650
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006333, upper bound: 0.0006389
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006333, upper bound: 0.0006397
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006407
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006432
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006637
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006657
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006637
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006657
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006319, upper bound: 0.0006387
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006319, upper bound: 0.0006397
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006407
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 1, lower bound: -0.0006407, upper bound: 0.0006432

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.29 + 596.97 = 600.26 seconds
